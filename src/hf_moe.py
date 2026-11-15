"""
Team NOVA - AFML Project
hf_moe.py: run our research pipeline on a PUBLIC pretrained Mixture-of-Experts model from Hugging Face.

Why: a result shown only on models we trained ourselves could be an artifact of our small models. We
repeat the experiments on IBM Granite-3.1-1B-A400M (1.3B parameters, 400M active, 24 MoE layers,
32 experts, top-8, trained on ~10T tokens, Apache-2.0) and on its official instruction-tuned version,
which gives REAL fine-tuning flips made by another team.

How: HFMoE wraps the Hugging Face model so it looks exactly like our NovaLM to the research code:
  model.tok_emb                 token embedding (hooked to add upstream noise)
  model.blocks[i].is_moe        True for the MoE layers we study (a subset, to save time)
  model.blocks[i].ffn           our MoE proxy: router = nn.Linear sharing the model's own router weight,
                                same top-k + softmax-over-top-k gates as the original, force_route hook,
                                and the model's own experts
  model.cfg                     top_k, n_experts, n_embd, n_layer, block_size (probe window length)
  model(idx, targets, ...)      -> (logits, loss, info) like NovaLM
The proxy computes EXACTLY what the original MoE block computes (tested on a random Granite model).

Load anywhere the research code takes a checkpoint path, with the prefix "hf:":
  hf:ibm-granite/granite-3.1-1b-a400m-base?layers=3,9,15,21&T=256
Make the probe data (our validation documents, re-tokenized with Granite's tokenizer):
  python hf_moe.py make_data --src_data_dir DATA --out_dir DATA_GRANITE --model ibm-granite/granite-3.1-1b-a400m-base
"""

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = "ibm-granite/granite-3.1-1b-a400m-base"


@dataclass
class HFConfig:
    top_k: int
    n_experts: int
    n_embd: int
    n_layer: int
    block_size: int
    vocab_size: int
    moe_layers: tuple = ()
    name: str = ""


class MoEProxy(nn.Module):
    """Drop-in replacement for a Hugging Face sparse MoE block (Granite / Mixtral style).
    router: nn.Linear(C, E) whose weight IS the original router weight (shared, not copied)."""

    def __init__(self, orig, top_k):
        super().__init__()
        w = orig.router.weight if hasattr(orig.router, "weight") else orig.router.layer.weight
        E, C = w.shape
        self.router = nn.Linear(C, E, bias=False)
        self.router.weight = w                              # share the parameter
        self.experts = orig.experts
        self.top_k = top_k
        self.force_route = None                             # same research hook as our MoEFFN
        self.last_routing = None

    def forward(self, layer_input):
        B, T, C = layer_input.shape
        h = layer_input.reshape(-1, C)
        logits = self.router(h).float()                     # (N, E), as in the original (float32)
        top_logits, top_idx = logits.topk(self.top_k, dim=-1)
        if self.force_route is not None:
            mask, forced = self.force_route
            top_idx = torch.where(mask[:, None], forced, top_idx)
            top_logits = logits.gather(1, top_idx)
        gates = top_logits.softmax(dim=-1).type_as(h)       # softmax over the top-k only, as the original
        self.last_routing = {"logits": logits.detach(), "top_idx": top_idx.detach()}
        out = self.experts(h, top_idx, gates)
        return out.view(B, T, C)


class _Block:
    def __init__(self, ffn, is_moe):
        self.ffn, self.is_moe = ffn, is_moe


class HFMoE(nn.Module):
    def __init__(self, hf_model, layers=None, block_size=256, name=""):
        super().__init__()
        self.hf = hf_model
        c = hf_model.config
        top_k = getattr(c, "num_experts_per_tok", None) or getattr(c, "top_k")
        n_exp = getattr(c, "num_local_experts", None) or getattr(c, "num_experts")
        dec = hf_model.model.layers
        self.proxies = nn.ModuleList()
        for layer in dec:
            attr = "block_sparse_moe" if hasattr(layer, "block_sparse_moe") else "mlp"
            proxy = MoEProxy(getattr(layer, attr), top_k)
            setattr(layer, attr, proxy)
            self.proxies.append(proxy)
        L = len(dec)
        layers = tuple(range(L)) if layers is None else tuple(layers)
        self.blocks = [_Block(p, i in layers) for i, p in enumerate(self.proxies)]
        self.tok_emb = hf_model.model.embed_tokens
        self.cfg = HFConfig(top_k=top_k, n_experts=n_exp, n_embd=c.hidden_size, n_layer=L,
                            block_size=block_size, vocab_size=c.vocab_size, moe_layers=layers, name=name)

    def num_params(self):
        return sum(p.numel() for p in self.hf.parameters())

    def num_active_params(self):
        total = self.num_params()
        per_expert = sum(p.numel() for p in self.proxies[0].experts.parameters()) // self.cfg.n_experts
        return total - len(self.proxies) * (self.cfg.n_experts - self.cfg.top_k) * per_expert

    def forward(self, idx, targets=None, loss_mask=None, return_routing=False, valid_mask=None):
        logits = self.hf(input_ids=idx, use_cache=False).logits
        info = {"aux_loss": None, "z_loss": None, "routing": {}}
        if return_routing:
            info["routing"] = {i: b.ffn.last_routing for i, b in enumerate(self.blocks) if b.is_moe}
        if targets is None:
            return logits, None, info
        per_token = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        if loss_mask is None:
            loss = per_token.mean()
        else:
            m = loss_mask.reshape(-1).float()
            loss = (per_token * m).sum() / m.sum().clamp(min=1.0)
        return logits, loss, info


def parse_spec(spec):
    """'hf:repo/name?layers=3,9&T=256' -> (name, layers or None, T)."""
    s = spec[3:] if spec.startswith("hf:") else spec
    name, _, query = s.partition("?")
    opts = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
    layers = tuple(int(x) for x in opts["layers"].split(",")) if "layers" in opts else None
    return name, layers, int(opts.get("T", 256))


def load_hf_moe(spec, device, dtype=torch.float32):
    """Load a public MoE as an HFMoE wrapper. float32 by default: the research needs exact second derivatives.
    attn_implementation='eager' and experts_implementation='eager': plain PyTorch code that supports
    double backward (the fused kernels do not)."""
    from transformers import AutoModelForCausalLM
    name, layers, T = parse_spec(spec)
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="eager",
                                              experts_implementation="eager")
    model = HFMoE(hf, layers=layers, block_size=T, name=name).to(device)
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Probe data: our validation documents re-tokenized with the public model's tokenizer
# ----------------------------------------------------------------------------
def make_data(src_data_dir, out_dir, model_name, max_docs=None):
    """Split our val.bin into documents (at <|endoftext|>), decode each with OUR tokenizer, encode with the
    public model's tokenizer, and write val.bin / val_source.bin / meta.json in our format. Document order is
    kept, so the development half (probe set) holds the same documents as ours."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(src_data_dir, "meta.json")))
    ours = Tokenizer.from_file(os.path.join(src_data_dir, "tokenizer.json"))
    theirs = AutoTokenizer.from_pretrained(model_name)
    eot_src = meta["special_tokens"]["<|endoftext|>"]
    val = np.fromfile(os.path.join(src_data_dir, "val.bin"), dtype=np.uint16)
    src_path = os.path.join(src_data_dir, "val_source.bin")
    src = np.fromfile(src_path, dtype=np.uint8) if os.path.exists(src_path) else np.zeros(len(val), np.uint8)
    ends = np.flatnonzero(val == eot_src)
    eos = theirs.eos_token_id
    ids_out, src_out, start = [], [], 0
    for n, e in enumerate(ends):
        if max_docs is not None and n >= max_docs:
            break
        text = ours.decode(val[start:e].tolist())
        ids = theirs(text, add_special_tokens=False)["input_ids"] + [eos]
        ids_out.append(np.asarray(ids, dtype=np.int64))
        src_out.append(np.full(len(ids), src[start], dtype=np.uint8))
        start = e + 1
    ids_all = np.concatenate(ids_out)
    dtype = np.uint16 if theirs.vocab_size <= 65535 and ids_all.max() < 65535 else np.uint32
    assert dtype == np.uint16, "the research code reads uint16 token files"
    ids_all.astype(np.uint16).tofile(os.path.join(out_dir, "val.bin"))
    np.concatenate(src_out).tofile(os.path.join(out_dir, "val_source.bin"))
    out_meta = {"dataset": f"re-tokenized validation of {meta.get('dataset', '?')} for {model_name}",
                "sources": meta.get("sources", ["data"]), "model": model_name,
                "vocab_size": len(theirs),
                "special_tokens": {"<|endoftext|>": eos, "<|pad|>": theirs.pad_token_id if theirs.pad_token_id is not None else eos},
                "val_tokens": int(ids_all.size), "val_stories": len(ids_out), "dtype": "uint16"}
    json.dump(out_meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    print(f"Wrote {ids_all.size:,} tokens ({len(ids_out):,} documents) to {out_dir}")
    return out_meta


@torch.no_grad()
def probe_perplexity(model, probe, device, batch=8):
    """Sanity check: next-token loss of the public model on the probe windows (should be low: it is a strong model)."""
    tot, n = 0.0, 0
    for s in range(0, len(probe), batch):
        w = torch.from_numpy(probe[s:s + batch]).to(device)
        loss = model(w[:, :-1], w[:, 1:])[1]
        tot += loss.item() * w[:, 1:].numel()
        n += w[:, 1:].numel()
    return tot / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["make_data"])
    p.add_argument("--src_data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max_docs", type=int, default=None)
    a = p.parse_args()
    make_data(a.src_data_dir, a.out_dir, a.model, a.max_docs)


if __name__ == "__main__":
    main()
