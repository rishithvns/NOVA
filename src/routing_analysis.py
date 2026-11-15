"""
Team NOVA - AFML Project
routing_analysis.py: measure how an MoE model routes tokens (Phase 2, reused in Phases 3-6).

For a FIXED probe set of validation tokens (the same tokens for every model and checkpoint),
it records, per MoE layer and per token:
  * router logits z (E values) and the chosen top-k experts
  * raw margin           z_(k) - z_(k+1)           gap between last chosen and first rejected expert
  * normalized margin    delta_min = min_{i in top-k, j not in top-k} (z_i - z_j) / ||w_i - w_j||
                         = exact L2 distance the router INPUT must move to flip the top-k set
                           (router is linear: z = W h, so the boundary z_i = z_j is a hyperplane)
  * routing entropy      H(softmax(z))
  * next-token loss      per-token cross-entropy (needed later for "does a flip matter?")

Outputs:
  <out_dir>/routing_<name>.npz   all per-token arrays (input to the flip-prediction experiments)
  <out_dir>/routing_summary.csv  one row per model x layer
  <out_dir>/expert_load.png, margin_hist.png
  <out_dir>/routing_by_source.csv, expert_by_source.png   (only for multi-dataset data: expert use per source)

Usage:
  python routing_analysis.py --data_dir DATA --out_dir RESULTS \
      --ckpts RUNS/moe/ckpt_last.pt RUNS/moe_nobal/ckpt_last.pt --names moe moe_nobal
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

from model import ModelConfig, NovaLM


# ----------------------------------------------------------------------------
# Probe set: fixed validation windows, created once and reused by every phase
# ----------------------------------------------------------------------------
def get_probe_set(data_dir, n_seq=64, T=256, seed=0):
    """Return (n_seq, T+1) token windows from the DEVELOPMENT half of val.bin (never the test half);
    saved to data_dir so it never changes."""
    from train import val_dev_end
    path = os.path.join(data_dir, f"probe_{n_seq}x{T}_dev.npy")
    if os.path.exists(path):
        return np.load(path)
    val_path = os.path.join(data_dir, "val.bin")
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    dev_end = val_dev_end(val_path, _eot(data_dir))
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, dev_end - T - 1, size=n_seq)
    probe = np.stack([data[i : i + T + 1] for i in ix]).astype(np.int64)
    np.save(path, probe)
    return probe


def probe_sources(data_dir, n_seq=64, T=256, seed=0):
    """Data source of every probe INPUT token, (n_seq, T) uint8, or None when the data is a single dataset.
    Recomputes the probe's window starts exactly as get_probe_set does (same seed, same development half)."""
    import json
    from train import val_dev_end
    src_path = os.path.join(data_dir, "val_source.bin")
    if not os.path.exists(src_path):
        return None, None
    names = json.load(open(os.path.join(data_dir, "meta.json")))["sources"]
    src = np.memmap(src_path, dtype=np.uint8, mode="r")
    dev_end = val_dev_end(os.path.join(data_dir, "val.bin"), _eot(data_dir))
    ix = np.random.default_rng(seed).integers(0, dev_end - T - 1, size=n_seq)
    return np.stack([src[i : i + T] for i in ix]), names


def by_source(name, res, src, names):
    """Expert load of each MoE layer separately for each data source, and how differently the sources use
    the experts: total-variation distance between a source's expert distribution and the overall one
    (0 = every source uses the experts identically, 1 = completely different experts)."""
    rows, E = [], int(res["n_experts"])
    s = src.reshape(-1)
    for i in res["moe_layers"]:
        top = res[f"L{i}_top_idx"]                                           # (N, k)
        overall = np.bincount(top.reshape(-1), minlength=E) / top.size
        for si, sname in enumerate(names):
            sel = top[s == si]
            if sel.size == 0:
                continue
            load = np.bincount(sel.reshape(-1), minlength=E) / sel.size
            row = {"model": name, "layer": int(i), "source": sname, "tokens": int((s == si).sum()),
                   "tv_distance_from_overall": round(float(0.5 * np.abs(load - overall).sum()), 4),
                   "top_expert": int(load.argmax()), "top_expert_share": round(float(load.max()), 4)}
            row.update({f"expert_{e}": round(float(load[e]), 4) for e in range(E)})
            rows.append(row)
    return rows


def plot_by_source(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(rows)
    keys = list(df.groupby(["model", "layer"]).groups)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.8 * len(keys), 2.6), squeeze=False)
    ecols = [c for c in df.columns if c.startswith("expert_")]
    for ax, (m, l) in zip(axes[0], keys):
        g = df[(df.model == m) & (df.layer == l)]
        im = ax.imshow(g[ecols].values, aspect="auto", cmap="Blues", vmin=0)
        ax.set_yticks(range(len(g))); ax.set_yticklabels(g.source, fontsize=8)
        ax.set_xticks(range(len(ecols))); ax.set_xticklabels(range(len(ecols)), fontsize=8)
        ax.set_xlabel("expert"); ax.set_title(f"{m} | layer {l}", fontsize=9)
    fig.suptitle("Share of routing choices per expert, by data source (different rows = specialization)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "expert_by_source.png"), dpi=150)


def _eot(data_dir):
    """<|endoftext|> id of this data folder (0 for our tokenizers; the public model's EOS for its probe data)."""
    import json
    try:
        return json.load(open(os.path.join(data_dir, "meta.json")))["special_tokens"]["<|endoftext|>"]
    except Exception:
        return 0


def load_model(ckpt_path, device):
    """Our checkpoints (.pt), or a public Hugging Face MoE with the prefix 'hf:' (see hf_moe.py)."""
    if str(ckpt_path).startswith("hf:"):
        from hf_moe import load_hf_moe
        return load_hf_moe(ckpt_path, device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = dict(ck["config"])
    c["moe_layers"] = tuple(c["moe_layers"])
    model = NovaLM(ModelConfig(**c)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Per-token routing features
# ----------------------------------------------------------------------------
def margins(logits, W, k):
    """
    logits: (N, E) router logits;  W: (E, C) router weight;  k: top-k.
    Returns raw_margin (N,), norm_margin (N,), entropy (N,).
    """
    z_sorted, order = logits.sort(dim=-1, descending=True)
    raw = z_sorted[:, k - 1] - z_sorted[:, k]

    top, rest = order[:, :k], order[:, k:]                                 # (N, k), (N, E-k)
    zi = logits.gather(1, top).unsqueeze(2)                                # (N, k, 1)
    zj = logits.gather(1, rest).unsqueeze(1)                               # (N, 1, E-k)
    dist = torch.cdist(W, W)                                               # (E, E): ||w_i - w_j||
    d = dist[top.unsqueeze(2), rest.unsqueeze(1)]                          # (N, k, E-k)
    norm = ((zi - zj) / d.clamp(min=1e-8)).flatten(1).min(dim=1).values    # (N,)

    p = logits.softmax(-1)
    ent = -(p * p.clamp(min=1e-12).log()).sum(-1)
    return raw, norm, ent


@torch.no_grad()
def collect(model, probe, device, batch=16):
    """Run the probe set through the model; return per-layer routing arrays + per-token loss."""
    k = model.cfg.top_k
    out = {"token_loss": [], "tokens": probe[:, :-1].reshape(-1)}
    layers = [i for i, b in enumerate(model.blocks) if b.is_moe]
    per = {i: {"logits": [], "top_idx": [], "raw_margin": [], "norm_margin": [], "entropy": []} for i in layers}
    for s in range(0, len(probe), batch):
        w = torch.from_numpy(probe[s : s + batch]).to(device)
        x, y = w[:, :-1], w[:, 1:]
        logits, _, info = model(x, return_routing=True)
        tl = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        out["token_loss"].append(tl.cpu())
        for i in layers:
            r = info["routing"][i]
            W = model.blocks[i].ffn.router.weight.float()
            raw, norm, ent = margins(r["logits"], W, k)
            per[i]["logits"].append(r["logits"].cpu())
            per[i]["top_idx"].append(r["top_idx"].cpu())
            per[i]["raw_margin"].append(raw.cpu())
            per[i]["norm_margin"].append(norm.cpu())
            per[i]["entropy"].append(ent.cpu())
    out["token_loss"] = torch.cat(out["token_loss"]).numpy()
    for i in layers:
        for key in per[i]:
            out[f"L{i}_{key}"] = torch.cat(per[i][key]).numpy()
    out["moe_layers"] = np.array(layers)
    out["n_experts"] = np.array(model.cfg.n_experts)
    out["top_k"] = np.array(k)
    return out


def summarize(name, res):
    rows = []
    E, k = int(res["n_experts"]), int(res["top_k"])
    for i in res["moe_layers"]:
        top = res[f"L{i}_top_idx"]
        load = np.bincount(top.reshape(-1), minlength=E) / top.size
        raw, norm = res[f"L{i}_raw_margin"], res[f"L{i}_norm_margin"]
        rows.append({
            "model": name, "layer": int(i),
            "max_load": round(float(load.max()), 4),          # 1/E = 0.125 is perfectly balanced
            "min_load": round(float(load.min()), 4),
            "load_cv": round(float(load.std() / load.mean()), 4),
            "mean_entropy": round(float(res[f"L{i}_entropy"].mean()), 4),   # max = ln(E)
            "raw_margin_median": round(float(np.median(raw)), 4),
            "norm_margin_median": round(float(np.median(norm)), 4),
            "frac_near_tie": round(float((raw < 0.1).mean()), 4),          # tokens close to flipping
            "mean_token_loss": round(float(res["token_loss"].mean()), 4),
        })
    return rows


def plot(all_res, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(all_res)
    # models may place their MoE layers differently (placement ablation): column c = each model's c-th MoE layer
    n_cols = max(len(all_res[n]["moe_layers"]) for n in names)
    E = int(all_res[names[0]]["n_experts"])

    fig, axes = plt.subplots(len(names), n_cols, figsize=(3.6 * n_cols, 2.8 * len(names)),
                             squeeze=False, sharey=True)
    for r, n in enumerate(names):
        for c, i in enumerate(all_res[n]["moe_layers"]):
            top = all_res[n][f"L{i}_top_idx"]
            load = np.bincount(top.reshape(-1), minlength=E) / top.size
            ax = axes[r][c]
            ax.bar(range(E), load, color="#4a78c2")
            ax.axhline(1 / E, ls="--", c="gray", lw=1)
            ax.set_title(f"{n} | layer {i}", fontsize=10)
            ax.set_xlabel("expert")
            if c == 0:
                ax.set_ylabel("fraction of routing slots")
    fig.suptitle("Expert load (dashed = perfectly balanced)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "expert_load.png"), dpi=150)

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 3), squeeze=False)
    for c in range(n_cols):
        ax = axes[0][c]
        for n in names:
            layers_n = list(all_res[n]["moe_layers"])
            if c < len(layers_n):
                i = layers_n[c]
                ax.hist(all_res[n][f"L{i}_norm_margin"], bins=60, histtype="step", density=True, label=f"{n} (layer {i})")
        ax.set_title(f"MoE layer #{c + 1} of each model")
        ax.set_xlabel("normalized margin (distance to flip)")
    axes[0][0].set_ylabel("density")
    axes[0][-1].legend()
    fig.suptitle("How far each token is from a routing flip")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "margin_hist.png"), dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--n_seq", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_res, rows = {}, []
    for ck, name in zip(args.ckpts, args.names):
        model = load_model(ck, args.device)
        probe = get_probe_set(args.data_dir, n_seq=args.n_seq, T=model.cfg.block_size)
        res = collect(model, probe, args.device)
        np.savez_compressed(os.path.join(args.out_dir, f"routing_{name}.npz"), **res)
        all_res[name] = res
        rows += summarize(name, res)
        print(f"{name}: {res['tokens'].size:,} probe tokens, MoE layers {list(res['moe_layers'])}")

    src_rows = []
    for name, res in all_res.items():
        T = int(len(res["tokens"]) // args.n_seq)
        src, names = probe_sources(args.data_dir, n_seq=args.n_seq, T=T)
        if src is not None:
            src_rows += by_source(name, res, src, names)
    if src_rows:
        with open(os.path.join(args.out_dir, "routing_by_source.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(src_rows[0]))
            w.writeheader()
            w.writerows(src_rows)
        plot_by_source(src_rows, args.out_dir)
        print("Per-source routing saved: routing_by_source.csv, expert_by_source.png")

    with open(os.path.join(args.out_dir, "routing_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    plot(all_res, args.out_dir)
    print(f"Saved results and plots to {args.out_dir}")


if __name__ == "__main__":
    main()
