"""
Team NOVA - AFML Project
experiments.py: the research experiments (Phases 4-7).

Research question: can LOCAL LOSS GEOMETRY predict which tokens' MoE routing is fragile,
beyond the routing margin, and do the predicted flips actually change the model's predictions?

Everything is measured on the SAME fixed probe set (64 validation windows x 256 tokens).
Expensive quantities are computed for a subsample of positions per window (default 32), for
every MoE layer. One row of the final table = one (window, position, layer).

PHASE 4 - geometry features of the PRETRAINED MoE model, per token t and MoE layer l:
  raw_margin     z_(k) - z_(k+1)                         gap last chosen / first rejected expert
  norm_margin    min_{i in top-k, j not} (z_i - z_j)/||w_i - w_j||   exact distance to flip in h-space
  entropy        H(softmax(z))
  token_loss     next-token cross-entropy L_t
  grad_h         ||dL_t / dh_t||                         loss gradient at the router input h_t
  margin_sens    ||d m_t / dE||                          margin gradient w.r.t. ALL token embeddings E
  lin_score      raw_margin / margin_sens                first-order "distance to flip" under embedding noise
  hess_top       top eigenvalue (by magnitude) of d^2 L_t / dh_t^2  (power iteration on Hessian-vector products)
  hess_trace     trace of d^2 L_t / dh_t^2               (Hutchinson estimator, Rademacher vectors)
  margin_curv    u^T (d^2 m_t / dE^2) u along the steepest margin-closing direction u: curvature of the ROUTING
                 decision in the SAME space as the upstream noise
  margin_hess_top top eigenvalue (by magnitude) of d^2 m_t / dE^2 (power iteration)
  quad_score     second-order distance to flip along u (smallest root of m - s a + c a^2 / 2 = 0)

PHASE 5 - flip labels from three perturbation sources:
  hidden_flip_p  router-input noise   h + eps  (sanity check: norm_margin should predict it almost perfectly)
  noise_flip_p   upstream noise       E + eps on every token embedding (the non-trivial case)
  ft_flip        instruction fine-tuning: top-k set before vs after fine-tuning

PHASE 6 - counterfactual consequences (does a flip matter?):
  dloss_near     L_t with the token forced onto its NEAREST alternative expert set, minus original L_t
  dloss_near_fut same, summed over the current and all later tokens of the window
  dloss_ft       L_t in the pretrained model forced onto the fine-tuned model's experts, minus original

PHASE 7 - training dynamics from saved snapshots: flip rate between consecutive snapshots,
  median margin, expert-load imbalance.

PHASE 8 - weight quantization (the setting of the closest prior work, arXiv 2608.11212): 4-bit round-to-nearest
  group quantization of all weights except embeddings and routers. Per token: quant_flip, quant_token_dloss, and
  quant_route_damage = loss with the quantized routing minus loss when forced back onto the clean routing
  (route-mediated damage). Analysis adds labels y_quant, y_quant_harmful, y_quant_damage_given_flip and a
  repair-at-budget curve: which share of the damage is removed by repairing the top-b% tokens chosen by each
  predictor / heuristic, vs random and an oracle.

Models: our NovaLM checkpoints, or a public Hugging Face MoE with the prefix "hf:" (see hf_moe.py).

The "copies trick": to get a per-token gradient / Hessian / counterfactual for P positions of one
window in ONE batched pass, the window is repeated P times and copy p only looks at position t_p.
Copies do not interact, so each copy's gradient is exactly that token's own gradient.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from model import ModelConfig, NovaLM
from routing_analysis import get_probe_set, load_model


# ----------------------------------------------------------------------------
# Forward pass with hooks: capture token embeddings E and router inputs h, optional noise on E
# ----------------------------------------------------------------------------
class Capture:
    def __init__(self, model):
        self.model, self.h, self.emb, self.emb_noise, self.handles = model, {}, None, None, []
        self.moe_layers = [i for i, b in enumerate(model.blocks) if b.is_moe]
        self.handles.append(model.tok_emb.register_forward_hook(self._emb_hook))
        for i in self.moe_layers:
            self.handles.append(model.blocks[i].ffn.register_forward_pre_hook(self._make_h_hook(i)))

    def _emb_hook(self, mod, inp, out):
        if self.emb_noise is not None:
            out = out + self.emb_noise
        self.emb = out
        return out

    def _make_h_hook(self, i):
        def hook(mod, inp):
            self.h[i] = inp[0]
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def router_logits(model, layer, h):
    return model.blocks[layer].ffn.router(h).float()


def token_losses(logits, y):
    B, T, V = logits.shape
    return F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), reduction="none").view(B, T)


def topk_sets(z, k):
    return z.topk(k, dim=-1).indices.sort(dim=-1).values          # sorted -> comparable as sets


def nearest_flip(z, W, k):
    """For each token: the expert pair (i out, j in) with the smallest normalized margin.
    Returns the alternative top-k set reached by the nearest flip, shape (..., k)."""
    shp = z.shape[:-1]
    z2 = z.reshape(-1, z.shape[-1])
    order = z2.argsort(dim=-1, descending=True)
    top, rest = order[:, :k], order[:, k:]
    dist = torch.cdist(W.float(), W.float())
    zi, zj = z2.gather(1, top).unsqueeze(2), z2.gather(1, rest).unsqueeze(1)
    ratio = (zi - zj) / dist[top.unsqueeze(2), rest.unsqueeze(1)].clamp(min=1e-8)   # (N, k, E-k)
    flat = ratio.flatten(1).argmin(dim=1)
    i_slot, j_slot = flat // rest.shape[1], flat % rest.shape[1]
    new = top.clone()
    new[torch.arange(len(new)), i_slot] = rest[torch.arange(len(new)), j_slot]
    return new.sort(dim=-1).values.view(*shp, k)


def subsample_positions(T, n_pos, start=16):
    return np.unique(np.linspace(start, T - 2, n_pos).round().astype(int))


# ----------------------------------------------------------------------------
# PHASE 4: geometry features
# ----------------------------------------------------------------------------
def math_attention():
    """Hessian-vector products need second derivatives through attention. The fused (flash /
    memory-efficient) attention kernels do not support double backward, so we switch PyTorch to
    its plain 'math' implementation: same result, slower, fully differentiable twice."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.MATH)


def geometry_features(model, probe, positions, device, n_power=10, n_hutch=5, seed=0, pos_chunk=0,
                      n_hutch_margin=4, n_sketch_margin=4):
    """Per (window, position, layer) geometry features via the copies trick. Model in fp32.
    pos_chunk > 0: process the positions in groups of this size (less GPU memory, same results,
    because copies never interact). Each group's copies are cut just after the group's last position:
    the model is causal, so later tokens cannot affect the loss or routing at earlier positions."""
    positions = np.asarray(positions)
    chunk = pos_chunk if pos_chunk and pos_chunk > 0 else len(positions)
    with math_attention():
        return _geometry_features(model, probe, positions, device, n_power, n_hutch, seed, chunk, n_hutch_margin,
                                  n_sketch_margin)


def _geometry_features(model, probe, positions, device, n_power, n_hutch, seed, chunk, n_hutch_margin=4,
                       n_sketch_margin=4):
    model.eval()
    k = model.cfg.top_k
    cap = Capture(model)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gen_m = torch.Generator(device="cpu").manual_seed(seed + 1)   # separate stream: other features unchanged
    rows = []
    for s in range(len(probe)):
        for c0 in range(0, len(positions), chunk):
            pos_c = positions[c0 : c0 + chunk]                     # this group of positions
            P = len(pos_c)
            pos_t = torch.as_tensor(pos_c, device=device)
            ar = torch.arange(P, device=device)
            T_used = int(pos_c.max()) + 1                          # causal: later tokens are irrelevant
            w = torch.from_numpy(probe[s]).to(device)
            x = w[:-1][:T_used].unsqueeze(0).repeat(P, 1)          # (P, T_used) copies of the window
            y = w[1:][:T_used].unsqueeze(0).repeat(P, 1)
            logits, _, _ = model(x)
            L = token_losses(logits, y)                                # (P, T)
            L_own = L[ar, pos_t]                                       # copy p -> its own position
            for layer in cap.moe_layers:
                h = cap.h[layer]                                       # (P, T, C)
                W = model.blocks[layer].ffn.router.weight
                z = router_logits(model, layer, h)                     # (P, T, E)
                z_own = z[ar, pos_t]                                   # (P, E)
                vals = z_own.topk(k + 1, dim=-1).values
                m = vals[:, k - 1] - vals[:, k]                        # raw margin (differentiable)

                # margin sensitivity to the embeddings of the whole window (first order)
                gE = torch.autograd.grad(m.sum(), cap.emb, create_graph=True)[0]          # (P, T, C)
                margin_sens = gE.detach().flatten(1).norm(dim=1)

                # margin CURVATURE in the same space as the upstream noise (second order).
                # u = steepest direction that closes the margin; c = u^T (d2 m / dE2) u
                def hvp_E(v):
                    return torch.autograd.grad((gE * v).sum(), cap.emb, retain_graph=True)[0]
                u = (-gE.detach() / margin_sens.clamp(min=1e-12)[:, None, None])
                margin_curv = (hvp_E(u) * u).flatten(1).sum(dim=1).detach()
                # top eigenvalue (by magnitude) of d2 m / dE2, power iteration
                ve = torch.randn(gE.shape, generator=gen).to(device)
                ve = ve / ve.flatten(1).norm(dim=1)[:, None, None]
                m_lam = torch.zeros(P, device=device)
                for _ in range(n_power):
                    Hv = hvp_E(ve)
                    m_lam = (Hv * ve).flatten(1).sum(dim=1)
                    ve = Hv / Hv.flatten(1).norm(dim=1).clamp(min=1e-12)[:, None, None]
                m_lam = m_lam.detach()
                # trace and squared Frobenius norm of H = d2m/dE2: the two numbers the second-order
                # flip-probability formula needs (flip_prob_formula). Hutch++ (Meyer et al., 2021):
                #   1. sketch: Q = orthonormal basis of H S, S = n_sketch Gaussian vectors (captures the dominant
                #      curvature directions; the margin Hessian typically has a +/- pair of large eigenvalues)
                #   2. exact part in that subspace: tr(Q^T H Q),  |H|_F^2 = 2|HQ|_F^2 - |Q^T H Q|_F^2 + |R|_F^2
                #   3. Hutchinson only on the remainder R = (I-QQ^T) H (I-QQ^T) with Gaussian probes r:
                #      E[r'^T H r'] = tr(R), E[|(I-QQ^T) H r'|^2] = |R|_F^2,   r' = (I-QQ^T) r
                # Unbiased; on our test Hessian 50x less noisy for |H|_F^2 and 5x for tr(H) than plain Hutchinson.
                D = gE[0].numel()
                Y = torch.stack([hvp_E(torch.randn(gE.shape, generator=gen_m).to(device)).detach().reshape(P, D)
                                 for _ in range(n_sketch_margin)], dim=2)                     # (P, D, s)
                Q = torch.linalg.qr(Y).Q                                                     # (P, D, s)
                HQ = torch.stack([hvp_E(Q[:, :, j].reshape(gE.shape)).detach().reshape(P, D)
                                  for j in range(Q.shape[2])], dim=2)                         # (P, D, s)
                Bm = Q.transpose(1, 2) @ HQ                                                  # (P, s, s)
                m_tr = torch.diagonal(Bm, dim1=1, dim2=2).sum(1)
                m_fro2 = 2 * HQ.pow(2).sum((1, 2)) - Bm.pow(2).sum((1, 2))
                proj = lambda X: X - Q @ (Q.transpose(1, 2) @ X)                              # (I - QQ^T) X
                for _ in range(n_hutch_margin):
                    r = proj(torch.randn((P, D, 1), generator=gen_m).to(device))
                    Hr = hvp_E(r.reshape(gE.shape)).detach().reshape(P, D, 1)
                    m_tr += (r * Hr).sum((1, 2)) / n_hutch_margin
                    m_fro2 += proj(Hr).pow(2).sum((1, 2)) / n_hutch_margin
                # second-order distance to flip along u: solve m - s*a + 0.5*c*a^2 = 0 for the smallest a > 0
                #   a = 2m / (s + sqrt(s^2 - 2cm)); if s^2 - 2cm < 0 the quadratic never reaches 0 -> capped at 10x first order
                md = m.detach()
                disc = margin_sens ** 2 - 2 * margin_curv * md
                quad = 2 * md / (margin_sens + disc.clamp(min=0).sqrt()).clamp(min=1e-12)
                quad = torch.where(disc >= 0, quad, 10 * md / margin_sens.clamp(min=1e-12))
                del gE

                # loss gradient and curvature at the router input h_t
                g_full = torch.autograd.grad(L_own.sum(), h, create_graph=True)[0]       # (P, T, C)
                g = g_full[ar, pos_t]                                                    # (P, C)
                grad_h = g.detach().norm(dim=1)

                def hvp(v):
                    return torch.autograd.grad((g * v).sum(), h, retain_graph=True)[0][ar, pos_t]

                v = torch.randn(g.shape, generator=gen).to(device)
                v = v / v.norm(dim=1, keepdim=True)
                lam = torch.zeros(P, device=device)
                for _ in range(n_power):
                    Hv = hvp(v)
                    lam = (Hv * v).sum(dim=1)                          # Rayleigh quotient
                    v = Hv / Hv.norm(dim=1, keepdim=True).clamp(min=1e-12)
                tr = torch.zeros(P, device=device)
                for _ in range(n_hutch):
                    r = (torch.randint(0, 2, g.shape, generator=gen) * 2 - 1).float().to(device)
                    tr += (hvp(r) * r).sum(dim=1) / n_hutch

                with torch.no_grad():
                    zd = z_own.detach()
                    srt, order = zd.sort(dim=-1, descending=True)
                    top, rest = order[:, :k], order[:, k:]
                    dist = torch.cdist(W.float(), W.float())
                    ratio = (zd.gather(1, top).unsqueeze(2) - zd.gather(1, rest).unsqueeze(1)) / \
                        dist[top.unsqueeze(2), rest.unsqueeze(1)].clamp(min=1e-8)
                    norm_margin = ratio.flatten(1).min(dim=1).values
                    p = zd.softmax(-1)
                    ent = -(p * p.clamp(min=1e-12).log()).sum(-1)
                for i in range(P):
                    rows.append({
                        "window": s, "pos": int(pos_c[i]), "layer": layer,
                        "raw_margin": m[i].item(), "norm_margin": norm_margin[i].item(),
                        "entropy": ent[i].item(), "token_loss": L_own[i].item(),
                        "grad_h": grad_h[i].item(), "margin_sens": margin_sens[i].item(),
                        "lin_score": (m[i] / margin_sens[i].clamp(min=1e-12)).item(),
                        "hess_top": lam[i].item(), "hess_trace": tr[i].item(),
                        "margin_curv": margin_curv[i].item(), "margin_hess_top": m_lam[i].item(),
                        "quad_score": quad[i].item(),
                        "margin_hess_trace": m_tr[i].item(), "margin_hess_fro2": m_fro2[i].item(),
                    })
                del g_full, g
            del logits, L
    cap.close()
    return rows


# ----------------------------------------------------------------------------
# PHASE 5: flip labels
# ----------------------------------------------------------------------------
@torch.no_grad()
def clean_routing(model, probe, device, batch=16):
    """Top-k sets and router logits for every probe token and MoE layer (no perturbation)."""
    model.eval()
    cap = Capture(model)
    sets, logits_all = {i: [] for i in cap.moe_layers}, {i: [] for i in cap.moe_layers}
    sq = {i: [] for i in cap.moe_layers}
    for s in range(0, len(probe), batch):
        x = torch.from_numpy(probe[s : s + batch, :-1]).to(device)
        model(x)
        for i in cap.moe_layers:
            sq[i].append(cap.h[i].float().pow(2).mean().item())
            z = router_logits(model, i, cap.h[i])
            logits_all[i].append(z.cpu())
            sets[i].append(topk_sets(z, model.cfg.top_k).cpu())
    cap.close()
    model.h_rms = {i: float(np.sqrt(np.mean(sq[i]))) for i in sq}     # RMS of router inputs, per layer
    return {i: torch.cat(sets[i]) for i in sets}, {i: torch.cat(logits_all[i]) for i in logits_all}


@torch.no_grad()
def embedding_rms(model, probe, device):
    x = torch.from_numpy(probe[:16, :-1]).to(device)
    return model.tok_emb(x).float().pow(2).mean().sqrt().item()


@torch.no_grad()
def upstream_noise_flips(model, probe, device, sigma, n_samples=8, batch=16, seed=0):
    """P(top-k set changes) per token when Gaussian noise (std sigma) is added to ALL embeddings."""
    model.eval()
    k = model.cfg.top_k
    base, _ = clean_routing(model, probe, device, batch)
    cap = Capture(model)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    flips = {i: torch.zeros(base[i].shape[:2]) for i in cap.moe_layers}
    for _ in range(n_samples):
        for s in range(0, len(probe), batch):
            x = torch.from_numpy(probe[s : s + batch, :-1]).to(device)
            B, T = x.shape
            cap.emb_noise = (torch.randn((B, T, model.cfg.n_embd), generator=gen) * sigma).to(device)
            model(x)
            for i in cap.moe_layers:
                sets = topk_sets(router_logits(model, i, cap.h[i]), k).cpu()
                flips[i][s : s + B] += (sets != base[i][s : s + B]).any(-1).float()
    cap.emb_noise = None
    cap.close()
    return {i: flips[i] / n_samples for i in flips}


@torch.no_grad()
def hidden_noise_flips(model, logits, sigma_rel, n_samples=8, seed=0):
    """Sanity check: noise added directly to the router input h. z' = z + W eps (router is linear).
    Noise std = sigma_rel x RMS of that layer's router inputs (model.h_rms, set by clean_routing)."""
    k = model.cfg.top_k
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for i, z in logits.items():
        W = model.blocks[i].ffn.router.weight.float().cpu()
        base = topk_sets(z, k)
        sigma = sigma_rel * model.h_rms[i]
        f = torch.zeros(z.shape[:2])
        for _ in range(n_samples):
            eps = torch.randn((*z.shape[:2], W.shape[1]), generator=gen) * sigma
            f += (topk_sets(z + eps @ W.T, k) != base).any(-1).float()
        out[i] = f / n_samples
    return out


@torch.no_grad()
def router_inputs(model, probe, device, batch=16):
    """Router input h (the RMSNorm output fed to each MoE layer) for every probe token, on CPU."""
    model.eval()
    cap = Capture(model)
    hs = {i: [] for i in cap.moe_layers}
    for s in range(0, len(probe), batch):
        model(torch.from_numpy(probe[s : s + batch, :-1]).to(device))
        for i in cap.moe_layers:
            hs[i].append(cap.h[i].float().cpu())
    cap.close()
    return {i: torch.cat(hs[i]) for i in hs}


FT_CAUSES = {0: "no flip", 1: "router input changed", 2: "router weights changed",
             3: "either change alone flips", 4: "only both changes together flip"}


@torch.no_grad()
def ft_flip_causes(pre, ft, probe, device):
    """Why did a token's experts change during fine-tuning?  Cross the pretrained and fine-tuned
    router INPUTS h with the pretrained and fine-tuned router WEIGHTS W (Wang et al. 2026 do this
    for model merging):
        flip            = top-k(W_ft h_ft)  != top-k(W_pre h_pre)
        input-driven    = top-k(W_pre h_ft) != top-k(W_pre h_pre)   (new input, old router)
        router-driven   = top-k(W_ft h_pre) != top-k(W_pre h_pre)   (old input, new router)
    Returns {layer: (n_seq, T) int codes, see FT_CAUSES}."""
    k = pre.cfg.top_k
    hp, hf = router_inputs(pre, probe, device), router_inputs(ft, probe, device)
    out = {}
    for i in hp:
        Wp = pre.blocks[i].ffn.router.weight.float().cpu()
        Wf = ft.blocks[i].ffn.router.weight.float().cpu()
        base = topk_sets(hp[i] @ Wp.T, k)
        flip = (topk_sets(hf[i] @ Wf.T, k) != base).any(-1)
        by_input = (topk_sets(hf[i] @ Wp.T, k) != base).any(-1)
        by_router = (topk_sets(hp[i] @ Wf.T, k) != base).any(-1)
        code = torch.zeros(flip.shape, dtype=torch.long)
        code[flip & by_input & ~by_router] = 1
        code[flip & ~by_input & by_router] = 2
        code[flip & by_input & by_router] = 3
        code[flip & ~by_input & ~by_router] = 4
        out[i] = code
    return out


# ----------------------------------------------------------------------------
# PHASE 6: counterfactual routing
# ----------------------------------------------------------------------------
@torch.no_grad()
def forced_route_losses(model, probe, positions, layer, forced_sets, device):
    """Force copy p of each window onto forced_sets[window, p] at (layer, positions[p]).
    Returns dloss_own (W, P) and dloss_future (W, P) relative to the unforced model."""
    model.eval()
    k, P = model.cfg.top_k, len(positions)
    ffn = model.blocks[layer].ffn
    pos_t = torch.as_tensor(positions, device=device)
    ar = torch.arange(P, device=device)
    d_own, d_fut = np.zeros((len(probe), P)), np.zeros((len(probe), P))
    for s in range(len(probe)):
        w = torch.from_numpy(probe[s]).to(device)
        x = w[:-1].unsqueeze(0).repeat(P, 1)
        y = w[1:].unsqueeze(0).repeat(P, 1)
        T = x.shape[1]
        L0 = token_losses(model(x)[0], y)            # (P, T) unforced, SAME batch shape as the forced run,
                                                     # so tiny shape-dependent float rounding cancels exactly
        mask = torch.zeros(P, T, dtype=torch.bool, device=device)
        mask[ar, pos_t] = True
        forced = torch.zeros(P, T, k, dtype=torch.long, device=device)
        forced[ar, pos_t] = torch.as_tensor(forced_sets[s], device=device)
        ffn.force_route = (mask.reshape(-1), forced.reshape(-1, k))
        try:
            L = token_losses(model(x)[0], y)                                  # (P, T)
        finally:
            ffn.force_route = None
        diff = L - L0
        d_own[s] = diff[ar, pos_t].cpu().numpy()
        after = torch.arange(T, device=device)[None] >= pos_t[:, None]
        d_fut[s] = (diff * after).sum(dim=1).cpu().numpy()
    return d_own, d_fut


@torch.no_grad()
def expert_ablation(model, probe, device, batch=16):
    """Disable one expert at a time (its router logit set to -inf, so its tokens go to the next-best
    expert) and measure the change in next-token loss: over all tokens, over the tokens that used
    that expert, and for those tokens split by routing margin (fragile vs confident).
    One-pager promise: 'measure the change in loss when experts are disabled'."""
    import pandas as pd

    model.eval()
    k = model.cfg.top_k
    base_sets, base_logits = clean_routing(model, probe, device, batch)

    def losses():
        out = []
        for s in range(0, len(probe), batch):
            w = torch.from_numpy(probe[s : s + batch]).to(device)
            out.append(token_losses(model(w[:, :-1])[0], w[:, 1:]).cpu())
        return torch.cat(out)

    L0 = losses()
    rows = []
    for i in base_sets:
        srt = base_logits[i].sort(dim=-1, descending=True).values
        margin = srt[..., k - 1] - srt[..., k]
        lo_cut, hi_cut = margin.quantile(0.25), margin.quantile(0.75)
        for e in range(model.cfg.n_experts):
            def mask_hook(mod, inp, out, e=e):
                out = out.clone()
                out[..., e] = torch.finfo(out.dtype).min
                return out
            h = model.blocks[i].ffn.router.register_forward_hook(mask_hook)
            try:
                d = losses() - L0
            finally:
                h.remove()
            used = (base_sets[i] == e).any(-1)
            frag, conf = used & (margin <= lo_cut), used & (margin >= hi_cut)
            rows.append({"layer": i, "expert": e, "share_tokens_using_expert": round(float(used.float().mean()), 4),
                         "dloss_all_tokens": round(float(d.mean()), 5),
                         "dloss_tokens_that_used_it": round(float(d[used].mean()), 5) if used.any() else float("nan"),
                         "dloss_used_fragile_margin": round(float(d[frag].mean()), 5) if frag.any() else float("nan"),
                         "dloss_used_confident_margin": round(float(d[conf].mean()), 5) if conf.any() else float("nan")})
    return pd.DataFrame(rows)


def min_upstream_flip(model, probe, positions, device, n_double=8, n_bisect=12, pos_chunk=0):
    """pos_chunk > 0: same computation on groups of positions, each with its copies cut after the
    group's last position (causal model, so identical results; much less GPU memory)."""
    positions = np.asarray(positions)
    chunk = pos_chunk if pos_chunk and pos_chunk > 0 else len(positions)
    rows = []
    for c0 in range(0, len(positions), chunk):
        rows += _min_upstream_flip(model, probe, positions[c0 : c0 + chunk], device, n_double, n_bisect)
    return rows


def _min_upstream_flip(model, probe, positions, device, n_double, n_bisect):
    """Empirical minimum embedding perturbation that flips a token's top-k set.
    Direction: minus the margin gradient w.r.t. all embeddings of the window (steepest way to close
    the gap). The first-order estimate of the distance is m / ||dm/dE|| (= lin_score). We then search
    the step size alpha for real: double until the top-k set changes, then bisect.
    actual / predicted = 1 means the model is locally linear; far from 1 = curvature matters.
    One-pager promise: 'determine the minimum perturbation required to change the selected top-k experts'."""
    model.eval()
    k = model.cfg.top_k
    cap = Capture(model)
    P = len(positions)
    pos_t = torch.as_tensor(positions, device=device)
    ar = torch.arange(P, device=device)
    rows = []
    for s in range(len(probe)):
        T_used = int(positions.max()) + 1                 # causal: later tokens cannot matter
        x = torch.from_numpy(probe[s, :-1][:T_used]).to(device).unsqueeze(0).repeat(P, 1)
        for layer in cap.moe_layers:
            with torch.enable_grad():
                cap.emb_noise = None
                model(x)
                z = router_logits(model, layer, cap.h[layer])[ar, pos_t]
                vals = z.topk(k + 1, dim=-1).values
                m = vals[:, k - 1] - vals[:, k]
                g = torch.autograd.grad(m.sum(), cap.emb)[0]                      # (P, T, C)
            gnorm = g.flatten(1).norm(dim=1).clamp(min=1e-12)
            direction = (-g / gnorm[:, None, None]).detach()
            pred = (m.detach() / gnorm)                                           # first-order distance
            base = topk_sets(z.detach(), k)

            def flipped(alpha):
                cap.emb_noise = direction * alpha[:, None, None]
                with torch.no_grad():
                    model(x)
                    zz = router_logits(model, layer, cap.h[layer])[ar, pos_t]
                return (topk_sets(zz, k) != base).any(-1)

            lo = torch.zeros(P, device=device)
            hi = pred.clamp(min=1e-4).clone()
            found = flipped(hi)
            for _ in range(n_double):                    # grow until every token flips (or give up)
                if found.all():
                    break
                lo = torch.where(found, lo, hi)
                hi = torch.where(found, hi, hi * 2)
                found = flipped(hi)
            for _ in range(n_bisect):
                mid = (lo + hi) / 2
                f = flipped(mid)
                hi = torch.where(f & found, mid, hi)
                lo = torch.where(f & found, lo, torch.where(found, mid, lo))
            for j in range(P):
                rows.append({"window": s, "pos": int(positions[j]), "layer": layer,
                             "min_up_dist": hi[j].item() if found[j] else float("nan"),
                             "min_up_pred": pred[j].item()})
    cap.emb_noise = None
    cap.close()
    return rows


# ----------------------------------------------------------------------------
# PHASE 8: weight quantization (the setting of the closest prior work, arXiv 2608.11212)
# ----------------------------------------------------------------------------
@torch.no_grad()
def fake_quantize_(model, bits=4, group=128, quant_router=False):
    """Simulated weight-only quantization, IN PLACE: symmetric round-to-nearest with one scale per group of
    `group` consecutive weights along the input dimension (the standard RTN / GPTQ-style grouping).
    Skipped: embeddings / output layer, norms (1-D), and the routers (unless quant_router), so routing flips
    come from upstream changes, the case our predictor targets. Returns the number of quantized weights."""
    qmax = 2 ** (bits - 1) - 1
    n = 0
    for name, p in model.named_parameters():
        if p.dim() < 2 or "tok_emb" in name or "embed" in name or "lm_head" in name:
            continue
        if "router" in name and not quant_router:
            continue
        g = group if p.shape[-1] % group == 0 else p.shape[-1]
        w = p.data.float().reshape(-1, g)
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
        q = (w / scale).round().clamp(-qmax - 1, qmax) * scale
        p.data.copy_(q.reshape(p.shape).to(p.dtype))
        n += p.numel()
    return n


@torch.no_grad()
def probe_token_losses(model, probe, device, batch=16):
    """Next-token loss of every probe token, (n_seq, T)."""
    model.eval()
    out = []
    for s in range(0, len(probe), batch):
        w = torch.from_numpy(probe[s : s + batch]).to(device)
        out.append(token_losses(model(w[:, :-1])[0], w[:, 1:]).cpu())
    return torch.cat(out)


def repair_curve(df, scores, damage_col, budgets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5), n_boot=1000, seed=0):
    """Repair-at-budget: if we may fix (restore the clean routing of) only a fraction b of the probe tokens,
    which share of the total route-mediated damage do we remove when we pick the tokens by each score?
    Damage per token = loss with its quantized routing minus loss with its clean routing (>0 = the flip hurt).
    'random' = expected value b; 'oracle' = picking by the true damage (upper bound). 95% CIs: bootstrap
    over probe windows (same resamples for every selector)."""
    d = df.reset_index(drop=True)
    dmg = np.clip(d[damage_col].fillna(0).values, 0, None)
    total = dmg.sum()
    rng = np.random.default_rng(seed)
    windows, g = np.unique(d["window"].values, return_inverse=True)
    W = np.stack([np.bincount(rng.integers(0, len(windows), len(windows)), minlength=len(windows))[g]
                  for _ in range(n_boot)]).astype(np.float32)
    rows = []
    sel = dict(scores)
    sel["oracle (true damage)"] = dmg
    for name, sc in sel.items():
        order = np.argsort(-np.asarray(sc, dtype=float), kind="stable")
        for b in budgets:
            k = max(1, int(round(b * len(d))))
            top = np.zeros(len(d), dtype=np.float32)
            top[order[:k]] = 1
            share = dmg[order[:k]].sum() / max(total, 1e-12)
            boot = (W * top * dmg).sum(1) / np.clip((W * dmg).sum(1), 1e-12, None)
            rows.append({"selector": name, "budget": b, "recovered_share": round(float(share), 4),
                         "ci_lo": round(float(np.percentile(boot, 2.5)), 4),
                         "ci_hi": round(float(np.percentile(boot, 97.5)), 4)})
    for b in budgets:
        rows.append({"selector": "random (expected)", "budget": b, "recovered_share": b, "ci_lo": b, "ci_hi": b})
    return rows


# ----------------------------------------------------------------------------
# PHASE 7: training dynamics
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Training-free, closed-form flip probability (second-order delta method)
# ----------------------------------------------------------------------------
def flip_prob_formula(m, g_norm, tr, fro2, sigma):
    """Probability that the routing margin m(E + eps) drops below 0 when every embedding coordinate gets
    independent Gaussian noise eps ~ N(0, sigma^2).
      first order : m(E+eps) ~ m + g.eps                 -> mean m, variance sigma^2 |g|^2
                    p1 = Phi(-m / (sigma |g|))
      second order: m(E+eps) ~ m + g.eps + eps.H.eps / 2 -> mean m + sigma^2 tr(H) / 2,
                    variance sigma^2 |g|^2 + sigma^4 |H|_F^2 / 2   (exact moments of the quadratic model)
                    p2 = Phi(-mean / sqrt(variance))      (normal approximation of its distribution)
    g = gradient of the margin w.r.t. all embeddings of the window (margin_sens = |g|), H = its Hessian
    (margin_hess_trace = tr H, margin_hess_fro2 = |H|_F^2). Returns (p1, p2)."""
    from scipy.stats import norm
    m, g_norm, tr, fro2 = (np.asarray(a, dtype=float) for a in (m, g_norm, tr, fro2))
    p1 = norm.cdf(-m / np.clip(sigma * g_norm, 1e-12, None))
    mean = m + 0.5 * sigma ** 2 * tr
    sd = np.sqrt(sigma ** 2 * g_norm ** 2 + 0.5 * sigma ** 4 * np.clip(fro2, 0, None))
    p2 = norm.cdf(-mean / np.clip(sd, 1e-12, None))
    return p1, p2


def analytic_flip_eval(df, rms, sigmas_rel, n_boot=1000, seed=0):
    """Compare the closed-form first- and second-order flip probabilities with the MEASURED flip rate under
    upstream noise (noise_flip_p@sigma, 16 samples per token). No training, no fitted parameters.
    Brier = mean squared difference between predicted and measured probability (lower = better);
    ECE = expected calibration error over 10 bins; Spearman = rank agreement; AUROC vs 'flips in >= half'.
    delta_brier = Brier(first order) - Brier(second order) > 0 means curvature makes the prediction better;
    95% CI and one-sided p from a paired bootstrap over windows (the same resamples for both)."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    rows, curves = [], {}
    if "margin_hess_trace" not in df:
        return rows, curves
    windows, g = np.unique(df["window"].values, return_inverse=True)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, len(windows), len(windows)), minlength=len(windows))[g]
                  for _ in range(n_boot)]).astype(np.float64)

    def ece(p, y, bins=10):
        idx = np.minimum((p * bins).astype(int), bins - 1)
        return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                         for b in range(bins) if (idx == b).any()))

    for rel in sigmas_rel:
        col = f"noise_flip_p@{rel}"
        if col not in df:
            continue
        y = df[col].values.astype(float)
        p1, p2 = flip_prob_formula(df["raw_margin"], df["margin_sens"], df["margin_hess_trace"],
                                   df["margin_hess_fro2"], rel * rms)
        curves[rel] = (y, p1, p2)
        e1, e2 = (p1 - y) ** 2, (p2 - y) ** 2
        b1 = (W * e1).sum(1) / W.sum(1)
        b2 = (W * e2).sum(1) / W.sum(1)
        d = b1 - b2
        yb = (y >= 0.5).astype(int)
        auc = (lambda p: round(float(roc_auc_score(yb, p)), 4)) if 0 < yb.sum() < len(yb) else (lambda p: float("nan"))
        rows.append({"sigma_rel": rel, "n": len(y), "measured_flip_rate": round(float(y.mean()), 4),
                     "brier_first": round(float(e1.mean()), 5), "brier_second": round(float(e2.mean()), 5),
                     "delta_brier": round(float(e1.mean() - e2.mean()), 5),
                     "delta_brier_lo": round(float(np.percentile(d, 2.5)), 5),
                     "delta_brier_hi": round(float(np.percentile(d, 97.5)), 5),
                     "p_second_not_better": round(float((np.sum(d <= 0) + 1) / (len(d) + 1)), 4),
                     "ece_first": round(ece(p1, y), 4), "ece_second": round(ece(p2, y), 4),
                     "spearman_first": round(float(spearmanr(p1, y)[0]), 4),
                     "spearman_second": round(float(spearmanr(p2, y)[0]), 4),
                     "auroc_first": auc(p1), "auroc_second": auc(p2)})
    return rows, curves


def snapshot_dynamics(run_dir, probe, device, pattern="snapshot_step*.pt"):
    files = sorted(glob.glob(os.path.join(run_dir, pattern)))
    rows, prev = [], None
    for f in files:
        model = load_model(f, device)
        step = torch.load(f, map_location="cpu", weights_only=False)["step"]
        sets, logits = clean_routing(model, probe, device)
        E, k = model.cfg.n_experts, model.cfg.top_k
        for i in sets:
            srt = logits[i].sort(dim=-1, descending=True).values
            load = torch.bincount(sets[i].reshape(-1), minlength=E).float()
            load = load / load.sum()
            rows.append({
                "run": os.path.basename(run_dir.rstrip("/")), "step": step, "layer": i,
                "flip_rate_vs_prev": float((sets[i] != prev[i]).any(-1).float().mean()) if prev else float("nan"),
                "median_raw_margin": float((srt[..., k - 1] - srt[..., k]).median()),
                "load_cv": float(load.std() / load.mean()),
            })
        prev = sets
    return rows


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def build_table(pre_ckpt, ft_ckpt, data_dir, out_dir, device, n_seq=64, n_pos=32,
                sigmas=(0.05, 0.1, 0.2, 0.4), n_samples=8, n_power=10, n_hutch=5, pos_chunk=0,
                quant_bits=4, quant_group=128, quant_router=False, n_hutch_margin=4):
    """Run Phases 4-6 and save one CSV row per (window, position, layer)."""
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    pre = load_model(pre_ckpt, device)
    T = pre.cfg.block_size
    probe = get_probe_set(data_dir, n_seq=n_seq, T=T)
    positions = subsample_positions(T, n_pos)
    k = pre.cfg.top_k
    moe_layers = [i for i, b in enumerate(pre.blocks) if b.is_moe]
    print(f"probe {probe.shape}, {len(positions)} positions/window, MoE layers {moe_layers}")

    print("Phase 4: geometry features ...")
    df = pd.DataFrame(geometry_features(pre, probe, positions, device, n_power, n_hutch, pos_chunk=pos_chunk,
                                        n_hutch_margin=n_hutch_margin))

    print("Phase 4b: minimum upstream perturbation that flips routing ...")
    mins = pd.DataFrame(min_upstream_flip(pre, probe, positions, device, pos_chunk=pos_chunk))
    df = df.merge(mins, on=["window", "pos", "layer"], how="left")
    df["min_up_ratio"] = df["min_up_dist"] / df["min_up_pred"]

    print("Phase 5: flip labels ...")
    base_sets, base_logits = clean_routing(pre, probe, device)
    rms = embedding_rms(pre, probe, device)
    pos_idx = torch.as_tensor(positions)
    flip_summary = []
    for rel in sigmas:
        up = upstream_noise_flips(pre, probe, device, sigma=rel * rms, n_samples=n_samples)
        hid = hidden_noise_flips(pre, base_logits, sigma_rel=rel, n_samples=n_samples)
        for i in moe_layers:
            a = up[i][:, pos_idx].numpy().reshape(-1)
            b = hid[i][:, pos_idx].numpy().reshape(-1)
            sel = df.layer == i
            df.loc[sel, f"noise_flip_p@{rel}"] = a
            df.loc[sel, f"hidden_flip_p@{rel}"] = b
            flip_summary.append({"sigma_rel": rel, "layer": i,
                                 "upstream_flip_rate_all_tokens": float(up[i].mean()),
                                 "hidden_flip_rate_all_tokens": float(hid[i].mean())})

    if ft_ckpt is not None:
        ft = load_model(ft_ckpt, device)
        ft_sets, _ = clean_routing(ft, probe, device)
        for i in moe_layers:
            changed = (ft_sets[i] != base_sets[i]).any(-1)
            df.loc[df.layer == i, "ft_flip"] = changed[:, pos_idx].numpy().reshape(-1).astype(int)
            flip_summary.append({"sigma_rel": "fine-tuning", "layer": i,
                                 "ft_flip_rate_all_tokens": float(changed.float().mean())})
        print("Phase 5b: why did fine-tuning flips happen? ...")
        causes = ft_flip_causes(pre, ft, probe, device)
        cause_rows = []
        for i in moe_layers:
            c = causes[i]
            df.loc[df.layer == i, "ft_cause"] = c[:, pos_idx].numpy().reshape(-1)
            flipped = c[c > 0]
            row = {"layer": i, "flipped_tokens": int(flipped.numel()),
                   "flip_rate_all_tokens": round(float((c > 0).float().mean()), 4)}
            for code in range(1, 5):
                row[f"share: {FT_CAUSES[code]}"] = round(float((flipped == code).float().mean()), 4) \
                    if flipped.numel() else float("nan")
            cause_rows.append(row)
        pd.DataFrame(cause_rows).to_csv(os.path.join(out_dir, "ft_flip_causes.csv"), index=False)
        del ft

    print("Phase 6: counterfactual routing ...")
    for i in moe_layers:
        W = pre.blocks[i].ffn.router.weight.detach()
        z_sel = base_logits[i][:, pos_idx].to(device)                       # (n_seq, P, E)
        near = nearest_flip(z_sel, W, k).cpu().numpy()
        d_own, d_fut = forced_route_losses(pre, probe, positions, i, near, device)
        sel = df.layer == i
        df.loc[sel, "dloss_near"] = d_own.reshape(-1)
        df.loc[sel, "dloss_near_fut"] = d_fut.reshape(-1)
        if ft_ckpt is not None:
            ftsel = ft_sets[i][:, pos_idx].numpy()
            d_ft, _ = forced_route_losses(pre, probe, positions, i, ftsel, device)
            df.loc[sel, "dloss_ft"] = d_ft.reshape(-1)

    print("Phase 6b: disabling experts one at a time ...")
    expert_ablation(pre, probe, device).to_csv(os.path.join(out_dir, "expert_ablation.csv"), index=False)

    if quant_bits:
        # LAST phase: quantizes the loaded model in place (it is not used afterwards)
        print(f"Phase 8: {quant_bits}-bit weight quantization: routing flips and route-mediated damage ...")
        clean_L = probe_token_losses(pre, probe, device)
        n_q = fake_quantize_(pre, bits=quant_bits, group=quant_group, quant_router=quant_router)
        q_sets, _ = clean_routing(pre, probe, device)
        q_L = probe_token_losses(pre, probe, device)
        for i in moe_layers:
            sel = df.layer == i
            changed = (q_sets[i] != base_sets[i]).any(-1)                        # (n_seq, T)
            df.loc[sel, "quant_flip"] = changed[:, pos_idx].numpy().reshape(-1).astype(int)
            df.loc[sel, "quant_token_dloss"] = (q_L - clean_L)[:, pos_idx].numpy().reshape(-1)
            # force the token back onto its CLEAN experts inside the quantized model:
            # damage = loss with quantized routing - loss with clean routing (0 where nothing flipped)
            clean_sel = base_sets[i][:, pos_idx].numpy()
            d_own, _ = forced_route_losses(pre, probe, positions, i, clean_sel, device)
            df.loc[sel, "quant_route_damage"] = (-d_own).reshape(-1)
            flip_summary.append({"sigma_rel": f"quant{quant_bits}", "layer": i,
                                 "quant_flip_rate_all_tokens": float(changed.float().mean()),
                                 "quant_mean_token_dloss": float((q_L - clean_L).mean())})
        print(f"  quantized {n_q:,} weights; mean loss change {float((q_L - clean_L).mean()):+.4f} nats")

    df.to_csv(os.path.join(out_dir, "token_table.csv"), index=False)
    pd.DataFrame(flip_summary).to_csv(os.path.join(out_dir, "flip_rates.csv"), index=False)
    json.dump({"pre_ckpt": pre_ckpt, "ft_ckpt": ft_ckpt, "n_seq": n_seq, "positions": positions.tolist(),
               "sigmas_rel": list(sigmas), "embedding_rms": rms, "n_samples": n_samples,
               "quant_bits": quant_bits, "quant_group": quant_group, "quant_router": quant_router},
              open(os.path.join(out_dir, "experiment_config.json"), "w"), indent=2)
    print(f"Saved {len(df)} rows to {out_dir}/token_table.csv")
    return df


# ----------------------------------------------------------------------------
# Statistics: can features predict flips?  (logistic regression, grouped CV, bootstrap CIs)
# ----------------------------------------------------------------------------
FIRST_ORDER = ["raw_margin", "norm_margin", "margin_sens", "lin_score", "grad_h"]
LOSS_CURV = ["hess_top", "hess_trace"]                         # curvature of the LOSS at the router input h
MARGIN_CURV = ["margin_curv", "margin_hess_top", "quad_score"]  # curvature of the MARGIN at the embeddings
FEATURE_SETS = {
    "raw margin": ["raw_margin"],
    "normalized margin": ["norm_margin"],
    "margins": ["raw_margin", "norm_margin"],
    # the ladder that tests the main hypothesis
    "margins + first-order": FIRST_ORDER,
    "margins + first-order + curvature": FIRST_ORDER + LOSS_CURV + MARGIN_CURV,
    # which kind of curvature helps?
    "first-order + loss curvature": FIRST_ORDER + LOSS_CURV,
    "first-order + margin curvature": FIRST_ORDER + MARGIN_CURV,
    "all features": FIRST_ORDER + LOSS_CURV + MARGIN_CURV + ["entropy", "token_loss"],
}


def _transform(X):
    """Signed log so heavy-tailed features (margins, curvature) work in a linear model."""
    return np.sign(X) * np.log1p(np.abs(X))


def boot_auc_ap(y, pred, W, chunk=50):
    """AUROC and AUPRC of `pred` under many bootstrap weightings at once (exact, vectorised).
    W[b, i] = how many times token i appears in resample b (resampling WINDOWS with replacement gives
    integer token weights). Identical to sklearn's roc_auc_score / average_precision_score on the
    resampled (duplicated) data, ties included, but ~100x faster."""
    order = np.argsort(-pred, kind="stable")
    p, ys = pred[order], y[order].astype(float)
    starts = np.r_[0, np.flatnonzero(np.diff(p)) + 1]          # groups of tied scores, highest first
    aucs, aps = [], []
    for c in range(0, len(W), chunk):
        w = W[c:c + chunk][:, order]
        pos = np.add.reduceat(w * ys, starts, axis=1)
        neg = np.add.reduceat(w * (1 - ys), starts, axis=1)
        tp, fp = np.cumsum(pos, axis=1), np.cumsum(neg, axis=1)
        Wp, Wn = tp[:, -1], fp[:, -1]
        aucs.append((pos * (Wn[:, None] - fp + 0.5 * neg)).sum(1) / (Wp * Wn))   # P(score_pos > score_neg)
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        aps.append((pos * prec).sum(1) / Wp)                                      # sum_k (R_k - R_k-1) P_k
    return np.concatenate(aucs), np.concatenate(aps)


def evaluate_predictors(df, label_col, n_boot=1000, seed=0, n_folds=5, classifier="logreg", return_preds=False,
                        feature_sets=None):
    """classifier: 'logreg' (pre-registered main analysis) or 'gbt' (gradient-boosted trees: a check that the
    conclusions do not depend on the linear model)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = df.dropna(subset=[label_col]).reset_index(drop=True)
    y = d[label_col].values.astype(int)
    groups = d["window"].values
    if y.sum() < 10 or (1 - y).sum() < 10:
        empty = [{"label": label_col, "features": "-", "note": f"too few positives ({y.sum()}/{len(y)})"}], []
        return (*empty, {}, d) if return_preds else empty
    rng = np.random.default_rng(seed)
    windows, g = np.unique(groups, return_inverse=True)
    # the SAME bootstrap resamples (over windows, not tokens) for every feature set -> paired tests.
    # Resample b draws windows with replacement; token i then has weight = count of its window in b.
    counts = np.stack([np.bincount(rng.integers(0, len(windows), len(windows)), minlength=len(windows))
                       for _ in range(n_boot)])
    W = counts[:, g].astype(np.float32)
    keep = ((W * y).sum(1) > 0) & ((W * (1 - y)).sum(1) > 0)        # resamples with both classes
    W = W[keep]
    out, preds, boot_scores = [], {}, {}
    sets = FEATURE_SETS if feature_sets is None else {k: FEATURE_SETS[k] for k in feature_sets}
    for name, cols in sets.items():
        X = _transform(d[cols].values.astype(float))
        X = np.hstack([X, np.eye(d.layer.nunique())[d.layer.astype("category").cat.codes]])  # layer one-hot
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups):
            if classifier == "gbt":
                from sklearn.ensemble import HistGradientBoostingClassifier
                clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                     l2_regularization=1.0, random_state=0)
            else:
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict_proba(X[te])[:, 1]
        preds[name] = pred
        aucs, aps = boot_auc_ap(y, pred, W)
        boot_scores[name] = (aucs, aps)
        out.append({"label": label_col, "features": name, "classifier": classifier, "n": len(y),
                    "positive_rate": round(y.mean(), 4),
                    "auroc": round(roc_auc_score(y, pred), 4),
                    "auroc_lo": round(np.percentile(aucs, 2.5), 4), "auroc_hi": round(np.percentile(aucs, 97.5), 4),
                    "auprc": round(average_precision_score(y, pred), 4),
                    "auprc_lo": round(np.percentile(aps, 2.5), 4), "auprc_hi": round(np.percentile(aps, 97.5), 4)})
    sig = paired_tests(y, preds, boot_scores, label_col) if all(a in preds and b in preds for a, b in SIG_COMPARISONS) else []
    return (out, sig, preds, d) if return_preds else (out, sig)


SIG_COMPARISONS = [
    # PRIMARY test (null hypothesis: curvature adds nothing once margin and first-order sensitivity are known)
    ("margins + first-order", "margins + first-order + curvature"),
    # secondary (Holm-corrected together with the primary)
    ("margins + first-order", "first-order + loss curvature"),
    ("margins + first-order", "first-order + margin curvature"),
    ("margins", "margins + first-order"),
    ("raw margin", "normalized margin"),
]


def paired_tests(y, preds, boot_scores, label_col):
    """Paired bootstrap test: is feature set B better than A on the same resampled windows?
    delta = AUROC(B) - AUROC(A) per resample; one-sided p = share of resamples with delta <= 0."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows = []
    for a, b in SIG_COMPARISONS:
        d_auc = boot_scores[b][0] - boot_scores[a][0]
        d_ap = boot_scores[b][1] - boot_scores[a][1]
        rows.append({"label": label_col, "baseline": a, "compared": b,
                     "delta_auroc": round(roc_auc_score(y, preds[b]) - roc_auc_score(y, preds[a]), 4),
                     "delta_auroc_lo": round(np.percentile(d_auc, 2.5), 4),
                     "delta_auroc_hi": round(np.percentile(d_auc, 97.5), 4),
                     "p_auroc": round((np.sum(d_auc <= 0) + 1) / (len(d_auc) + 1), 4),
                     "delta_auprc": round(average_precision_score(y, preds[b]) - average_precision_score(y, preds[a]), 4),
                     "p_auprc": round((np.sum(d_ap <= 0) + 1) / (len(d_ap) + 1), 4)})
    # Holm-Bonferroni correction over the comparisons for this label (controls false positives across tests)
    order = np.argsort([r["p_auroc"] for r in rows])
    m, running = len(rows), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[i]["p_auroc"]))
        rows[i]["p_auroc_holm"] = round(running, 4)
    for r in rows:
        r["primary"] = r["baseline"] == SIG_COMPARISONS[0][0] and r["compared"] == SIG_COMPARISONS[0][1]
    return rows


def cross_strength(df, n_folds=5):
    """Train on flip labels at one upstream-noise strength, test on held-out windows at EVERY other
    strength. Does the predictor generalize to perturbation sizes it never saw?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cols = sorted([c for c in df.columns if c.startswith("noise_flip_p@")], key=lambda c: float(c.split("@")[1]))
    groups = df["window"].values
    layer_oh = np.eye(df.layer.nunique())[df.layer.astype("category").cat.codes]
    rows = []
    for name in ["margins", "margins + first-order", "margins + first-order + curvature"]:
        X = np.hstack([_transform(df[FEATURE_SETS[name]].values.astype(float)), layer_oh])
        for c_tr in cols:
            y_tr = (df[c_tr] >= 0.5).astype(int).values
            if y_tr.sum() < 10 or (1 - y_tr).sum() < 10:
                continue
            pred = np.zeros(len(df))
            for tr, te in GroupKFold(n_splits=n_folds).split(X, y_tr, groups):
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                clf.fit(X[tr], y_tr[tr])
                pred[te] = clf.predict_proba(X[te])[:, 1]
            for c_te in cols:
                y_te = (df[c_te] >= 0.5).astype(int).values
                if y_te.min() == y_te.max():
                    continue
                rows.append({"features": name, "train_sigma": float(c_tr.split("@")[1]),
                             "test_sigma": float(c_te.split("@")[1]), "auroc": round(roc_auc_score(y_te, pred), 4)})
    return rows


def analyze(out_dir, harmful_threshold=0.1, target_flip_rate=0.15):
    """Build binary labels, run all predictors, save results.csv, correlations.csv and plots."""
    import pandas as pd
    from scipy.stats import spearmanr

    df = pd.read_csv(os.path.join(out_dir, "token_table.csv"))
    labels = {}
    # pick the upstream-noise level whose mean flip probability is closest to the target rate
    noise_cols = [c for c in df.columns if c.startswith("noise_flip_p@")]
    best = min(noise_cols, key=lambda c: abs(df[c].mean() - target_flip_rate))
    df["y_noise"] = (df[best] >= 0.5).astype(int)
    labels["y_noise"] = f"upstream noise ({best.split('@')[1]} x embedding RMS), flip in >= half the samples"
    hid = best.replace("noise_flip_p", "hidden_flip_p")
    df["y_hidden"] = (df[hid] >= 0.5).astype(int)
    labels["y_hidden"] = "router-input noise (sanity check)"
    if "ft_flip" in df:
        df["y_ft"] = df["ft_flip"]
        labels["y_ft"] = "routing changed during instruction fine-tuning"
    if "ft_cause" in df:
        df["y_ft_input"] = (df["ft_cause"] == 1).astype(int)
        labels["y_ft_input"] = "fine-tuning flip caused by the router INPUT changing (router weights alone would not flip)"
    df["y_harmful"] = (df["dloss_near"] > harmful_threshold).astype(int)
    labels["y_harmful"] = f"nearest flip raises own loss by > {harmful_threshold} nats"
    if df["y_harmful"].mean() < 0.05:            # too rare to study: use the most damaging 20% instead
        cut = df["dloss_near"].quantile(0.8)
        df["y_harmful"] = (df["dloss_near"] > cut).astype(int)
        labels["y_harmful"] = f"nearest flip in the top 20% by loss increase (> {cut:.4f} nats)"

    # PRIMARY target: consequential instability = the token flips under upstream noise AND its nearest
    # flip clearly hurts its next-token prediction
    df["y_consequential"] = ((df["y_noise"] == 1) & (df["y_harmful"] == 1)).astype(int)
    labels["y_consequential"] = "consequential instability: flips under upstream noise AND the flip is harmful"
    if "quant_flip" in df:
        df["y_quant"] = df["quant_flip"].astype(int)
        labels["y_quant"] = "routing changes under 4-bit weight quantization"
        df["y_quant_harmful"] = ((df["quant_flip"] == 1) & (df["quant_route_damage"] > 0)).astype(int)
        labels["y_quant_harmful"] = "flips under quantization AND restoring the clean routing lowers the loss (route-mediated damage)"
        df["y_quant_damage_given_flip"] = np.where(df["quant_flip"] == 1, (df["quant_route_damage"] > 0).astype(float), np.nan)
        labels["y_quant_damage_given_flip"] = ("among tokens that flipped under quantization: harmful (1) vs harmless (0); "
                                               "the comparison where the router margin failed in arXiv 2608.11212 (AUC 0.49)")
    labels = {"y_consequential": labels.pop("y_consequential"), **labels}       # primary first

    results, sig = [], []
    for col in labels:
        r, sg = evaluate_predictors(df, col)
        results += r
        sig += sg
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    pd.DataFrame(sig).to_csv(os.path.join(out_dir, "significance.csv"), index=False)
    pd.DataFrame(cross_strength(df)).to_csv(os.path.join(out_dir, "cross_strength.csv"), index=False)

    # robustness to the classifier: the same ladder with gradient-boosted trees
    ladder = ["margins", "margins + first-order", "margins + first-order + curvature", "all features"]
    gbt = []
    for col in [c for c in ["y_consequential", "y_noise", "y_harmful", "y_quant_harmful", "y_quant_damage_given_flip"]
                if c in labels]:
        gbt += evaluate_predictors(df, col, classifier="gbt", feature_sets=ladder)[0]
    pd.DataFrame(gbt).to_csv(os.path.join(out_dir, "results_gbt.csv"), index=False)

    # repair at a budget (quantization): learned predictors (out-of-fold) and simple training-free heuristics
    if "quant_route_damage" in df:
        _, _, preds, dq = evaluate_predictors(df, "y_quant_harmful", return_preds=True,
                                              feature_sets=["raw margin", "margins", "margins + first-order",
                                                            "margins + first-order + curvature", "all features"])
        # learned selectors need >= 10 harmful flips; the training-free heuristics are always reported
        scores = {f"learned: {k}": v for k, v in preds.items()}
        scores.update({"heuristic: smallest raw margin": -dq["raw_margin"].values,
                       "heuristic: smallest first-order flip distance": -dq["lin_score"].values,
                       "heuristic: smallest second-order flip distance": -dq["quad_score"].values})
        pd.DataFrame(repair_curve(dq, scores, "quant_route_damage")).to_csv(
            os.path.join(out_dir, "repair_curve.csv"), index=False)

    feats = FEATURE_SETS["all features"]
    targets = [best, "min_up_dist", "dloss_near", "dloss_near_fut"] + (["dloss_ft"] if "dloss_ft" in df else [])
    corr = pd.DataFrame([{"feature": f, **{t: round(spearmanr(df[f], df[t], nan_policy="omit")[0], 3)
                                          for t in targets}} for f in feats])
    corr.to_csv(os.path.join(out_dir, "correlations.csv"), index=False)
    json.dump(labels, open(os.path.join(out_dir, "labels.json"), "w"), indent=2)

    # training-free closed-form flip probability (first vs second order) against the measured flip rates
    cfg_path = os.path.join(out_dir, "experiment_config.json")
    if os.path.exists(cfg_path) and "margin_hess_trace" in df:
        cfg = json.load(open(cfg_path))
        an_rows, curves = analytic_flip_eval(df, cfg["embedding_rms"], cfg["sigmas_rel"])
        if an_rows:
            pd.DataFrame(an_rows).to_csv(os.path.join(out_dir, "analytic_flip.csv"), index=False)
            plot_calibration(curves, out_dir, float(best.split("@")[1]))
    plot_results(df, res, labels, out_dir, best)
    return df, res, corr, labels


def plot_calibration(curves, out_dir, main_sigma, bins=10):
    """Reliability diagram: predicted flip probability (closed form) vs measured flip rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rels = sorted(curves)
    fig, axes = plt.subplots(1, len(rels), figsize=(3.4 * len(rels), 3.3), squeeze=False, sharey=True)
    for ax, rel in zip(axes[0], rels):
        y, p1, p2 = curves[rel]
        for p, lab, mk in ((p1, "first order", "o"), (p2, "second order (+ curvature)", "s")):
            idx = np.minimum((p * bins).astype(int), bins - 1)
            xs = [p[idx == b].mean() for b in range(bins) if (idx == b).sum() >= 5]
            ys = [y[idx == b].mean() for b in range(bins) if (idx == b).sum() >= 5]
            ax.plot(xs, ys, marker=mk, label=lab)
        ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
        ax.set_title(f"noise {rel} x RMS" + (" (main)" if rel == main_sigma else ""), fontsize=9)
        ax.set_xlabel("predicted flip probability")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("measured flip rate")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Closed-form flip probability vs measured (no training)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration_analytic.png"), dpi=150)


def plot_results(df, res, labels, out_dir, noise_col):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. AUROC per feature set per label, with bootstrap CIs
    r = res[res.get("auroc").notna()] if "auroc" in res else res
    labs = [l for l in labels if l in set(r.label)]
    sets = list(FEATURE_SETS)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.8 / max(len(labs), 1)
    for j, l in enumerate(labs):
        sub = r[r.label == l].set_index("features").reindex(sets)
        xs = np.arange(len(sets)) + (j - (len(labs) - 1) / 2) * width
        short = {"y_consequential": "consequential (flip + harmful)",
                 "y_noise": "upstream noise", "y_hidden": "router-input noise (sanity check)",
                 "y_ft": "fine-tuning", "y_ft_input": "input-driven fine-tuning flip",
                 "y_harmful": "harmful flip", "y_quant": "quantization flip",
                 "y_quant_harmful": "harmful quantization flip",
                 "y_quant_damage_given_flip": "harmful vs harmless (given a quantization flip)"}.get(l, l)
        ax.bar(xs, sub.auroc, width, label=short,
               yerr=[sub.auroc - sub.auroc_lo, sub.auroc_hi - sub.auroc], capsize=2)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([x.replace(" + ", "\n+ ") for x in sets], fontsize=8)
    ax.set_ylabel("AUROC (grouped 5-fold CV)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Which features predict a routing flip?", pad=40)
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "auroc_by_feature_set.png"), dpi=150)

    # 2. flip probability vs linearized score and vs raw margin
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, f in zip(axes, ["raw_margin", "lin_score"]):
        q = df[f].rank(pct=True)
        bins = np.linspace(0, 1, 11)
        mids = 0.5 * (bins[1:] + bins[:-1])
        means = [df[noise_col][(q > a) & (q <= b)].mean() for a, b in zip(bins[:-1], bins[1:])]
        ax.plot(mids * 100, means, marker="o")
        ax.set_xlabel(f"{f} percentile")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("upstream-noise flip probability")
    fig.suptitle("Flip probability by feature decile")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "flip_prob_by_decile.png"), dpi=150)

    # 3. first-order prediction vs actual minimum upstream flip distance
    if "min_up_dist" in df:
        d = df.dropna(subset=["min_up_dist"])
        d = d[(d.min_up_pred > 0) & (d.min_up_dist > 0)]
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        for layer, g in d.groupby("layer"):
            ax.scatter(g.min_up_pred, g.min_up_dist, s=5, alpha=0.4, label=f"layer {layer}")
        lim = [d[["min_up_pred", "min_up_dist"]].min().min(), d[["min_up_pred", "min_up_dist"]].max().max()]
        ax.plot(lim, lim, c="gray", lw=1, ls="--", label="actual = predicted")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("predicted (first order): margin / ||dm/dE||")
        ax.set_ylabel("actual minimum flip distance")
        ax.set_title("Is routing locally linear?")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "min_flip_distance.png"), dpi=150)

    # 3b. repair at a budget (quantization)
    rc_path = os.path.join(out_dir, "repair_curve.csv")
    if os.path.exists(rc_path):
        import pandas as pd
        rc = pd.read_csv(rc_path)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for name, g in rc.groupby("selector", sort=False):
            style = dict(ls="--", c="gray") if name.startswith("random") else (dict(ls=":", c="black") if name.startswith("oracle") else {})
            ax.plot(g.budget * 100, g.recovered_share * 100, marker="o", ms=3, label=name, **style)
        ax.set_xlabel("tokens repaired (% of probe tokens, chosen by the selector)")
        ax.set_ylabel("route-mediated damage removed (%)")
        ax.set_title("Repair at a budget under 4-bit quantization")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "repair_curve.png"), dpi=150)

    # 4. how much does the nearest flip hurt?
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(df["dloss_near"].clip(-2, 2), bins=80)
    ax.axvline(0, c="gray", lw=1)
    ax.set_xlabel("change in own next-token loss when forced to nearest flip (nats)")
    ax.set_ylabel("tokens")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dloss_near_hist.png"), dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["table", "analyze", "dynamics"])
    p.add_argument("--data_dir")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pre_ckpt")
    p.add_argument("--ft_ckpt", default=None)
    p.add_argument("--run_dirs", nargs="*", default=[])
    p.add_argument("--n_seq", type=int, default=64)
    p.add_argument("--n_pos", type=int, default=32)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--n_power", type=int, default=10)
    p.add_argument("--n_hutch", type=int, default=5)
    p.add_argument("--n_hutch_margin", type=int, default=4,
                   help="Hutch++ probes (after a 4-vector sketch) for the trace and Frobenius norm of the margin Hessian")
    p.add_argument("--quant_bits", type=int, default=4, help="weight-quantization phase (0 = skip)")
    p.add_argument("--quant_group", type=int, default=128)
    p.add_argument("--quant_router", action="store_true", help="also quantize the routers")
    p.add_argument("--pos_chunk", type=int, default=0,
                   help="positions per batched pass in the gradient phases (0 = all; 8 for the Base model on a T4)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.command == "table":
        build_table(a.pre_ckpt, a.ft_ckpt, a.data_dir, a.out_dir, a.device, a.n_seq, a.n_pos,
                    n_samples=a.n_samples, n_power=a.n_power, n_hutch=a.n_hutch, pos_chunk=a.pos_chunk,
                    quant_bits=a.quant_bits, quant_group=a.quant_group, quant_router=a.quant_router,
                    n_hutch_margin=a.n_hutch_margin)
    elif a.command == "analyze":
        _, res, corr, labels = analyze(a.out_dir)
        import pandas as pd
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(res)
            print(pd.read_csv(os.path.join(a.out_dir, "significance.csv")))
            print(corr)
    else:
        import pandas as pd
        rows = []
        for rd in a.run_dirs:
            probe = get_probe_set(a.data_dir, n_seq=a.n_seq, T=load_model(
                sorted(glob.glob(os.path.join(rd, "snapshot_step*.pt")))[0], "cpu").cfg.block_size)
            rows += snapshot_dynamics(rd, probe, a.device)
        pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "dynamics.csv"), index=False)
        print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
