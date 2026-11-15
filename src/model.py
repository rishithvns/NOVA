"""
Team NOVA - AFML Project
model.py: our own decoder-only transformer, with an optional Mixture-of-Experts FFN.

Differences from the video (GPT-2 style) model, and why:
  * RMSNorm instead of LayerNorm  - no mean-centering, fewer ops, equally stable (LLaMA)
  * RoPE instead of learned position embeddings - relative positions via rotating Q and K
  * SwiGLU feed-forward instead of GELU MLP     - gated FFN, lower loss at equal params
  * No bias terms                                - fewer params, no quality loss at this scale
  * Weight tying (kept from GPT-2)               - input embedding = output projection
  * MoE switch: any layer listed in config.moe_layers uses MoEFFN instead of SwiGLU.
    With moe_layers = () the model is a plain dense transformer (Phase 1).
  * Optional (used by the Base model on C4, off for the small model):
      - Grouped-query attention (n_kv_head < n_head): several query heads share one key/value head
        (LLaMA-2 70B, LLaMA-3, Mistral). Fewer parameters and a 3x smaller KV cache, same quality.
      - QK-norm (qk_norm=True): RMSNorm on each query and key head before RoPE (OLMo-2, Qwen3,
        Gemma-3). Keeps attention logits bounded, which matters in float16 on a T4.

Shapes: B = batch, T = sequence length, C = n_embd, H = n_head, D = head size, V = vocab.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    block_size: int = 256          # max context length T
    n_layer: int = 6
    n_head: int = 6                # query heads H
    n_kv_head: int = 0             # key/value heads; 0 means = n_head (ordinary multi-head attention)
    qk_norm: bool = False          # RMSNorm on q and k per head
    n_embd: int = 384              # C
    ffn_hidden: int = 1024         # SwiGLU hidden size (~ 8/3 * C, rounded)
    dropout: float = 0.0           # each training token is seen about once, so no dropout
    rope_theta: float = 10000.0
    # --- Mixture of Experts (Phase 2). Empty tuple = dense model. ---
    moe_layers: tuple = ()         # e.g. (1, 3, 5) = MoE in alternate layers
    n_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 512       # each expert is half the dense FFN width


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * g.  Like LayerNorm but without subtracting the mean."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # compute in float32 for stability under mixed precision
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def build_rope_cache(T, D, theta, device):
    """Precompute cos/sin for rotary embeddings. Returns two (T, D/2) tensors."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, device=device).float() / D))  # (D/2,)
    t = torch.arange(T, device=device).float()                                       # (T,)
    freqs = torch.outer(t, inv_freq)                                                 # (T, D/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """Rotate each pair (x1, x2) of features by an angle that depends on the position.

    x: (B, H, T, D). Pairing: first half of D with second half.
    After rotation, q·k depends only on the RELATIVE distance between positions.
    """
    D = x.shape[-1]
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    cos, sin = cos[None, None], sin[None, None]          # (1, 1, T, D/2)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head or cfg.n_head                    # Hkv
        assert self.n_head % self.n_kv_head == 0, "query heads must split evenly into KV groups"
        self.head_dim = cfg.n_embd // cfg.n_head                        # D
        self.kv_dim = self.n_kv_head * self.head_dim                    # Hkv * D (= C without GQA)
        self.qkv = nn.Linear(cfg.n_embd, cfg.n_embd + 2 * self.kv_dim, bias=False)  # C -> C + 2*Hkv*D
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)      # C -> C
        if cfg.qk_norm:
            self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split([C, self.kv_dim, self.kv_dim], dim=2)  # (B,T,C), 2 x (B,T,Hkv*D)
        # (B, T, C) -> (B, H, T, D); keys and values have Hkv heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:                                     # QK-norm, before RoPE
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.n_kv_head != self.n_head:                               # GQA: each KV head serves
            rep = self.n_head // self.n_kv_head                         # n_head / n_kv_head query heads
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        # softmax(q k^T / sqrt(D) + causal mask) v, computed by a fused kernel
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )                                                               # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                # (B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    """FFN(x) = W_down( SiLU(W_gate x) * (W_up x) ).  Applied to each token independently."""

    def __init__(self, n_embd, hidden):
        super().__init__()
        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.up = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """Top-k Mixture of Experts feed-forward layer (Phase 2).

    router:  logits z = W_r h            (one score per expert, per token)
    select:  the top_k experts by z
    combine: y = sum_{i in top-k} g_i * Expert_i(h),  g = softmax over the top-k logits

    Also returns:
      aux_loss  - Switch-Transformer load-balancing loss  E * sum_i f_i * P_i
      z_loss    - mean(logsumexp(z)^2), keeps router logits small and stable
      routing   - dict with logits / chosen experts, used by the research analysis
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.n_embd, cfg.expert_hidden) for _ in range(cfg.n_experts)]
        )
        # Research hook (Phase 6, counterfactual routing). Not a parameter, never saved.
        # Set to (mask, idx): mask (N,) bool marks tokens to override, idx (N, k) their forced experts.
        self.force_route = None

    def forward(self, x, valid=None):
        """valid: optional (B, T) bool mask; padding tokens (False) are left out of the balancing
        statistics (aux loss, z-loss). They are still routed, which does not affect real tokens
        because causal attention never lets a real token see the padding after it."""
        B, T, C = x.shape
        h = x.reshape(-1, C)                                            # (N, C), N = B*T
        logits = self.router(h).float()                                 # (N, E)
        probs = logits.softmax(dim=-1)                                  # (N, E)
        top_logits, top_idx = logits.topk(self.top_k, dim=-1)           # (N, k)
        if self.force_route is not None:                                # counterfactual: force experts
            mask, forced = self.force_route
            top_idx = torch.where(mask[:, None], forced, top_idx)
            top_logits = logits.gather(1, top_idx)                      # gates from the forced experts
        gates = top_logits.softmax(dim=-1).type_as(h)                   # (N, k), renormalized

        out = torch.zeros_like(h)
        for e in range(self.n_experts):
            token_pos, slot = (top_idx == e).nonzero(as_tuple=True)     # tokens routed to e
            if token_pos.numel() == 0:
                continue
            y = self.experts[e](h[token_pos])                           # (n_e, C)
            out.index_add_(0, token_pos, y * gates[token_pos, slot, None])

        # load balancing: f_i = fraction of routing slots sent to expert i,
        #                 P_i = mean router probability of expert i
        v = torch.ones(h.shape[0], device=h.device) if valid is None else valid.reshape(-1).float()
        n_valid = v.sum().clamp(min=1.0)
        f = torch.zeros(self.n_experts, device=h.device)
        f.scatter_add_(0, top_idx.reshape(-1), v.repeat_interleave(self.top_k))
        f = f / (n_valid * self.top_k)
        P = (probs * v[:, None]).sum(dim=0) / n_valid
        aux_loss = self.n_experts * (f * P).sum()
        z_loss = (torch.logsumexp(logits, dim=-1).pow(2) * v).sum() / n_valid

        routing = {"logits": logits.detach(), "top_idx": top_idx.detach()}
        return out.view(B, T, C), aux_loss, z_loss, routing


class Block(nn.Module):
    """Pre-norm transformer block:  x = x + attn(norm(x));  x = x + ffn(norm(x))."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.is_moe = layer_idx in cfg.moe_layers
        self.ffn = MoEFFN(cfg) if self.is_moe else SwiGLU(cfg.n_embd, cfg.ffn_hidden)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, valid=None):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin))
        if self.is_moe:
            y, aux, z, routing = self.ffn(self.norm2(x), valid)
            return x + self.drop(y), aux, z, routing
        return x + self.drop(self.ffn(self.norm2(x))), None, None, None


# ----------------------------------------------------------------------------
# The full model
# ----------------------------------------------------------------------------
class NovaLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight                      # weight tying

        cos, sin = build_rope_cache(cfg.block_size, cfg.n_embd // cfg.n_head, cfg.rope_theta, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scale down the layers that write into the residual stream (GPT-2 trick):
        # each block adds 2 contributions, so std / sqrt(2 * n_layer) keeps variance steady
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        """Total parameters (tied weights counted once)."""
        return sum(p.numel() for p in self.parameters())

    def num_active_params(self):
        """Parameters used per token (MoE: only top_k of n_experts experts)."""
        total = self.num_params()
        for b in self.blocks:
            if b.is_moe:
                per_expert = sum(p.numel() for p in b.ffn.experts[0].parameters())
                total -= (self.cfg.n_experts - self.cfg.top_k) * per_expert
        return total

    def forward(self, idx, targets=None, loss_mask=None, return_routing=False, valid_mask=None):
        """
        idx:       (B, T) token ids
        targets:   (B, T) next-token ids (idx shifted by one) -> teacher forcing
        loss_mask: (B, T) 1 where the loss counts, 0 elsewhere (instruction tuning, Phase 3)
        valid_mask: (B, T) True for real tokens, False for padding (MoE balancing statistics only)
        Returns logits (B, T, V), loss (or None), and an info dict.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        x = self.tok_emb(idx)                                           # (B, T, C)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]

        aux_losses, z_losses, routings = [], [], {}
        for i, block in enumerate(self.blocks):
            x, aux, z, routing = block(x, cos, sin, valid_mask)
            if aux is not None:
                aux_losses.append(aux)
                z_losses.append(z)
                if return_routing:
                    routings[i] = routing
        x = self.norm_f(x)
        logits = self.lm_head(x)                                        # (B, T, V)

        info = {
            "aux_loss": torch.stack(aux_losses).mean() if aux_losses else None,
            "z_loss": torch.stack(z_losses).mean() if z_losses else None,
            "routing": routings,
        }
        if targets is None:
            return logits, None, info

        per_token = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        )
        if loss_mask is None:
            loss = per_token.mean()
        else:
            m = loss_mask.reshape(-1).float()
            loss = (per_token * m).sum() / m.sum().clamp(min=1.0)
        return logits, loss, info

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, eot_id=None):
        """Sample tokens one at a time. Stops early when every sequence has produced eot_id."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            next_id = torch.multinomial(logits.softmax(-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eot_id is not None and (next_id == eot_id).all():
                break
        return idx
