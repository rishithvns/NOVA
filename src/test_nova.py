"""
Team NOVA - AFML Project
test_nova.py: correctness tests for the model, training utilities and research math.

Run:  python test_nova.py        (CPU is fine, about 1-2 minutes; no data or GPU needed)

Every test uses small randomly initialized models and synthetic tokens, and compares our fast
implementation against a slow, obviously-correct reference (brute force, finite differences,
or a manual loop).
"""

import math
import os
import tempfile

import numpy as np
import pandas as pd
import json
import torch
import torch.nn.functional as F

import experiments as ex
from model import ModelConfig, NovaLM
import train
from train import Batcher, val_dev_end

torch.manual_seed(0)
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def tiny(moe=True, **kw):
    cfg = dict(vocab_size=300, block_size=32, n_layer=2, n_head=2, n_embd=32, ffn_hidden=64,
               expert_hidden=32, n_experts=4, top_k=2, moe_layers=(1,) if moe else ())
    cfg.update(kw)
    return NovaLM(ModelConfig(**cfg)).eval()


# ------------------------------------------------------------------ model
def test_param_counts():
    d, m = NovaLM(ModelConfig()), NovaLM(ModelConfig(moe_layers=(1, 3, 5)))
    check("parameter counts (dense 13,767,552; MoE 24,393,600 total / 13,776,768 active)",
          d.num_params() == 13_767_552 and m.num_params() == 24_393_600 and m.num_active_params() == 13_776_768,
          f"{d.num_params():,} / {m.num_params():,} / {m.num_active_params():,}")


def test_base_presets():
    V = 16384
    size = lambda name: NovaLM(ModelConfig(vocab_size=V, **train.PRESETS[name]))
    d, m, d60, m60 = size("base_dense"), size("base_moe"), size("base60_dense"), size("base60_moe")
    ok = (d.num_params() == 88_101_120 and m.num_params() == 130_586_880 and m.num_active_params() == 88_119_552
          and abs(m.num_active_params() / d.num_params() - 1) < 0.001
          and d60.num_params() == 60_060_800 and m60.num_params() == 89_936_000 and m60.num_active_params() == 60_076_160
          and train.PRESETS["base_moe"] == train.PRESETS["base_moe_nobal"]
          and train.PRESETS["base60_moe"] == train.PRESETS["base60_moe_nobal"]
          and len({tuple(sorted(v.items())) for v in train.PRESET_TRAIN.values()}) == 1)
    check("Base presets (C4): dense 88,101,120; MoE 130,586,880 total / 88,119,552 active (within 0.1% of dense); "
          "fallback 60M presets unchanged; all Base runs share one training recipe", ok,
          f"{d.num_params():,} / {m.num_params():,} / {m.num_active_params():,}")


def test_balancing_switch():
    on = [p for p in train.PRESETS if train.uses_balancing(p)]
    off = [p for p in train.PRESETS if not train.uses_balancing(p)]
    check("only the *_nobal ablation presets train without the balancing loss",
          sorted(off) == ["base60_moe_nobal", "base_moe_nobal", "moe_nobal"] and "base_moe" in on, f"off: {sorted(off)}")


ADV = dict(n_head=4, n_kv_head=2, qk_norm=True)       # grouped-query attention + QK-norm, as in the Base model


def test_gqa_qknorm_attention():
    from model import apply_rope, RMSNorm
    m = tiny(**ADV); a = m.blocks[0].attn
    x = torch.randn(2, 10, 32)
    cos, sin = m.rope_cos[:10], m.rope_sin[:10]
    out = a(x, cos, sin)
    # slow reference: loop over query heads; query head h reads key/value head h // (H / Hkv)
    W = a.qkv.weight; Wq, Wk, Wv = W[:32], W[32:48], W[48:64]
    D, ref = 8, []
    for h in range(4):
        g = h // 2
        q = (x @ Wq[h * D:(h + 1) * D].T)[:, None]; k = (x @ Wk[g * D:(g + 1) * D].T)[:, None]
        v = (x @ Wv[g * D:(g + 1) * D].T)
        q, k = apply_rope(a.q_norm(q), cos, sin)[:, 0], apply_rope(a.k_norm(k), cos, sin)[:, 0]
        s = q @ k.transpose(1, 2) / math.sqrt(D) + torch.full((10, 10), float("-inf")).triu(1)
        ref.append(s.softmax(-1) @ v)
    ref = a.proj(torch.cat(ref, -1))
    m2 = tiny(n_head=4); same = tiny(n_head=4, n_kv_head=4); same.load_state_dict(m2.state_dict())
    xx = torch.randint(0, 300, (2, 32))
    check("grouped-query attention + QK-norm = per-head loop reference; n_kv_head = n_head is plain attention; "
          "QK-norm rows have unit RMS",
          torch.allclose(out, ref, atol=1e-5) and torch.allclose(m2(xx)[0], same(xx)[0], atol=1e-6)
          and torch.allclose(a.k_norm(torch.randn(5, 8)).pow(2).mean(-1), torch.ones(5), atol=1e-4),
          f"max diff {(out - ref).abs().max():.1e}")
    mc = tiny(**ADV); xc = torch.randint(0, 300, (2, 32)); xc2 = xc.clone(); xc2[:, 20] = (xc2[:, 20] + 5) % 300
    check("causal mask also holds with grouped-query attention", torch.allclose(mc(xc)[0][:, :20], mc(xc2)[0][:, :20], atol=1e-5))


def test_initial_loss():
    m = tiny()
    x = torch.randint(0, 300, (4, 32)); y = torch.randint(0, 300, (4, 32))
    loss = m(x, y)[1].item()
    check("initial loss close to ln(V) (uniform guessing)", abs(loss - math.log(300)) < 0.15, f"{loss:.3f} vs {math.log(300):.3f}")


def test_causal():
    m = tiny()
    x = torch.randint(0, 300, (2, 32)); x2 = x.clone(); x2[:, 20] = (x2[:, 20] + 5) % 300
    a, b = m(x)[0], m(x2)[0]
    check("causal mask: changing token 20 leaves positions < 20 unchanged",
          torch.allclose(a[:, :20], b[:, :20], atol=1e-5) and not torch.allclose(a[:, 20:], b[:, 20:]))


def test_rope_relative():
    from model import apply_rope, build_rope_cache
    cos, sin = build_rope_cache(64, 16, 10000.0, "cpu")
    q, k = torch.randn(16), torch.randn(16)
    rot = lambda v, p: apply_rope(v.view(1, 1, 1, 16).expand(1, 1, 64, 16), cos, sin)[0, 0, p]
    a, b = rot(q, 10) @ rot(k, 7), rot(q, 40) @ rot(k, 37)
    check("RoPE: q·k depends only on relative position", torch.allclose(a, b, atol=1e-5), f"{a:.5f} = {b:.5f}")


def test_moe_matches_manual():
    m = tiny(); moe = m.blocks[1].ffn
    h = torch.randn(2, 5, 32)
    out = moe(h)[0].reshape(-1, 32)
    hf = h.reshape(-1, 32); z = moe.router(hf); tl, ti = z.topk(2, -1); g = tl.softmax(-1)
    ref = torch.stack([sum(g[n, j] * moe.experts[ti[n, j]](hf[n:n + 1])[0] for j in range(2)) for n in range(len(hf))])
    check("MoE layer equals a slow manual top-2 computation", torch.allclose(out, ref, atol=1e-5))


def test_aux_loss_and_padding():
    m = tiny(); moe = m.blocks[1].ffn
    h = torch.randn(1, 10, 32)
    _, aux, z, r = moe(h)
    logits = r["logits"]; probs = logits.softmax(-1); ti = r["top_idx"]
    f = torch.bincount(ti.reshape(-1), minlength=4).float() / ti.numel()
    old_aux = 4 * (f * probs.mean(0)).sum()
    old_z = torch.logsumexp(logits, -1).pow(2).mean()
    same = torch.allclose(aux, old_aux, atol=1e-6) and torch.allclose(z, old_z, atol=1e-6)
    valid = torch.zeros(1, 10, dtype=torch.bool); valid[:, :6] = True
    _, aux_m, z_m, _ = moe(h, valid)
    _, aux_s, z_s, _ = moe(h[:, :6])
    check("load-balancing loss: unchanged without a mask; padding excluded with a mask",
          same and torch.allclose(aux_m, aux_s, atol=1e-6) and torch.allclose(z_m, z_s, atol=1e-6))


def test_loss_mask():
    m = tiny()
    x = torch.randint(0, 300, (3, 32)); y = torch.randint(0, 300, (3, 32))
    mask = torch.zeros(3, 32); mask[:, 16:] = 1
    lm = m(x, y, loss_mask=mask)[1]
    ref = F.cross_entropy(m(x)[0][:, 16:].reshape(-1, 300), y[:, 16:].reshape(-1))
    check("loss mask: only masked (response) positions count", torch.allclose(lm, ref, atol=1e-5))


def test_learns():
    m = tiny(); m.train()
    x = torch.randint(0, 300, (4, 32)); y = torch.randint(0, 300, (4, 32))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        _, loss, info = m(x, y)
        (loss + 0.01 * info["aux_loss"]).backward(); opt.step(); opt.zero_grad()
        first = first or loss.item()
    check("gradients reach every part: the model memorizes one batch", loss.item() < 0.2, f"{first:.2f} -> {loss.item():.3f}")


# ------------------------------------------------------------------ data split
def test_dev_test_split():
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    toks = []
    for _ in range(400):
        toks += list(rng.integers(4, 300, rng.integers(20, 60))) + [0]
    np.asarray(toks, dtype=np.uint16).tofile(os.path.join(d, "val.bin"))
    p = os.path.join(d, "val.bin")
    end = val_dev_end(p, 0)
    data = np.memmap(p, dtype=np.uint16, mode="r")
    from routing_analysis import get_probe_set
    probe = get_probe_set(d, n_seq=50, T=32)
    starts = [int(np.where((np.lib.stride_tricks.sliding_window_view(data, 33) == w).all(1))[0][0]) for w in probe[:10]]
    b = Batcher(p, 32, 8, "cpu", 0, limit=end)
    check("validation split: test half starts after a story end; probe and monitoring stay in the first half",
          data[end - 1] == 0 and end >= len(data) // 2 and max(starts) + 33 <= end and b.n == end)


# ------------------------------------------------------------------ research math
def setup_research(adv=False):
    m = tiny(**(ADV if adv else {}))
    probe = np.random.default_rng(1).integers(4, 300, (3, 33)).astype(np.int64)
    pos = ex.subsample_positions(32, 6, start=6)
    return m, probe, pos


def test_closed_form_margin():
    m = tiny(); r = m.blocks[1].ffn.router; W = r.weight.detach()
    h = torch.randn(500, 32); z = r(h).detach()
    _, norm, _ = __import__("routing_analysis").margins(z, W, 2)
    up = dn = rnd = 0
    for n in range(500):
        top = z[n].topk(2).indices.tolist(); rest = [j for j in range(4) if j not in top]
        d, i, j = min(((z[n, i] - z[n, j]) / (W[i] - W[j]).norm(), i, j) for i in top for j in rest)
        u = (W[j] - W[i]) / (W[j] - W[i]).norm()
        t0 = set(top)
        up += set(((h[n] + u * d * 1.01) @ W.T).topk(2).indices.tolist()) != t0
        dn += set(((h[n] + u * d * 0.99) @ W.T).topk(2).indices.tolist()) != t0
        v = torch.randn(32); v = v / v.norm() * norm[n] * 0.99
        rnd += set(((h[n] + v) @ W.T).topk(2).indices.tolist()) != t0
    check("closed-form flip distance is exact (flips 500/500 just past it, 0/500 just before)",
          up == 500 and dn == 0 and rnd == 0, f"{up}/500, {dn}/500, random {rnd}/500")


def test_geometry_against_brute_force(adv=False):
    pre = "[GQA + QK-norm model] " if adv else ""
    m, probe, pos = setup_research(adv)
    rows = ex.geometry_features(m, probe[:1], pos, "cpu", n_power=300, n_hutch=3000)
    r = [x for x in rows if x["layer"] == 1][2]
    t = int(pos[2]); w = torch.from_numpy(probe[0]); x = w[:-1][None]; y = w[1:][None]
    cap = ex.Capture(m)
    with ex.math_attention():
        # independent reference: gradient and Hessian-vector product w.r.t. a leaf perturbation of the embeddings
        def margin_at(delta, model=m):
            cap.emb_noise = delta
            model(x)
            v = model.blocks[1].ffn.router(cap.h[1])[0, t].topk(3).values
            return v[1] - v[2]
        m(x); d = torch.zeros_like(cap.emb.detach(), requires_grad=True)
        g = torch.autograd.grad(margin_at(d), d, create_graph=True)[0]
        u = (-g / g.norm()).detach()
        c_ref = (torch.autograd.grad((g * u).sum(), d)[0] * u).sum().item()
        cap.emb_noise = None
        m(x); h0 = cap.h[1].detach().clone()

        def Lt_of(ht):
            def hook(mod, inp):
                hh = inp[0].clone(); hh[0, t] = ht; return (hh,)
            hd = m.blocks[1].ffn.register_forward_pre_hook(hook)
            lg = m(x)[0]; hd.remove()
            return F.cross_entropy(lg[0, t], y[0, t])
        H = torch.autograd.functional.hessian(Lt_of, h0[0, t])
    cap.close()
    ev = torch.linalg.eigvalsh(H); top = ev[ev.abs().argmax()].item()
    off = (H ** 2).sum() - (torch.diagonal(H) ** 2).sum()
    se = math.sqrt(2 * off.item() / 3000)       # Hutchinson standard error with 3000 samples
    check(pre + "per-token margin gradient (copies trick) = direct gradient",
          abs(r["margin_sens"] - g.norm().item()) < 1e-4 * g.norm().item(), f"{r['margin_sens']:.5f} vs {g.norm():.5f}")
    check(pre + "loss-Hessian top eigenvalue (power iteration) = exact Hessian",
          abs(r["hess_top"] - top) < 0.02 * abs(top) + 1e-7, f"{r['hess_top']:.6f} vs {top:.6f}")
    check(pre + "loss-Hessian trace (Hutchinson) ≈ exact trace, within 4 standard errors",
          abs(r["hess_trace"] - H.trace().item()) < 4 * se + 1e-9, f"{r['hess_trace']:.6f} vs {H.trace():.6f}, s.e. {se:.6f}")
    check(pre + "margin curvature (copies trick) = independent Hessian-vector product",
          abs(r["margin_curv"] - c_ref) < 1e-3 * abs(c_ref) + 1e-5, f"{r['margin_curv']:.5f} vs {c_ref:.5f}")

    # finite-difference check of the curvature in 64-bit precision (32-bit rounding swamps tiny steps)
    md = tiny(**(ADV if adv else {})); md.load_state_dict(m.state_dict()); md = md.double()
    from model import RMSNorm
    for mod in md.modules():          # RMSNorm normally computes in float32 for training stability; use 64-bit here
        if isinstance(mod, RMSNorm):
            mod.forward = (lambda z, mod=mod: z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + mod.eps) * mod.weight)
    cap = ex.Capture(md)
    with ex.math_attention():
        def margin64(delta):
            cap.emb_noise = delta
            md(x)
            v = md.blocks[1].ffn.router(cap.h[1])[0, t].topk(3).values
            return v[1] - v[2]
        md(x); d = torch.zeros_like(cap.emb.detach(), requires_grad=True)
        m0 = margin64(d)
        g = torch.autograd.grad(m0, d, create_graph=True)[0]
        u = (-g / g.norm()).detach()
        c64 = (torch.autograd.grad((g * u).sum(), d)[0] * u).sum().item()
        a = 0.05 * m0.item() / g.norm().item()              # small, and far inside the margin
        with torch.no_grad():
            zero = torch.zeros_like(u)
            fd = ((margin64(a * u) + margin64(-a * u) - 2 * margin64(zero)) / a ** 2).item()
    cap.close()
    check(pre + "margin curvature = finite-difference second derivative (64-bit)", abs(c64 - fd) < 1e-3 * abs(c64) + 1e-6,
          f"{c64:.6f} vs {fd:.6f}")


def test_geometry_advanced_attention():
    test_geometry_against_brute_force(adv=True)


def test_position_chunking():
    m, probe, pos = setup_research(adv=True)
    full = ex.geometry_features(m, probe[:2], pos, "cpu", n_power=30, n_hutch=2)
    part = ex.geometry_features(m, probe[:2], pos, "cpu", n_power=30, n_hutch=2, pos_chunk=2)
    key = lambda r: (r["window"], r["pos"], r["layer"])
    full, part = sorted(full, key=key), sorted(part, key=key)
    exact = ["raw_margin", "norm_margin", "entropy", "token_loss", "grad_h", "margin_sens", "margin_curv", "quad_score"]
    err = max(abs(a[c] - b[c]) / (abs(a[c]) + 1e-6) for a, b in zip(full, part) for c in exact)
    mf = sorted(ex.min_upstream_flip(m, probe[:2], pos, "cpu"), key=key)
    mp = sorted(ex.min_upstream_flip(m, probe[:2], pos, "cpu", pos_chunk=2), key=key)
    same_min = all((np.isnan(a["min_up_dist"]) and np.isnan(b["min_up_dist"])) or
                   abs(a["min_up_dist"] - b["min_up_dist"]) <= 1e-4 * abs(a["min_up_dist"]) + 1e-7 for a, b in zip(mf, mp))
    check("position chunking + causal truncation give the same (non-random) features and flip distances (memory saver for Base)",
          len(full) == len(part) and err < 1e-4 and same_min, f"max rel. diff {err:.1e} on the deterministic features")


def test_counterfactuals_and_ablation():
    m, probe, pos = setup_research()
    sets, logits = ex.clean_routing(m, probe, "cpu")
    own = sets[1][:, torch.as_tensor(pos)].numpy()
    d_own, d_fut = ex.forced_route_losses(m, probe, pos, 1, own, "cpu")
    near = ex.nearest_flip(logits[1][:, torch.as_tensor(pos)], m.blocks[1].ffn.router.weight.detach(), 2).numpy()
    one_diff = bool(((near[..., :, None] == own[..., None, :]).any(-1).sum(-1) == 1).all())
    hook = m.blocks[1].ffn.router.register_forward_hook(
        lambda mod, i, o: o.clone().index_fill_(-1, torch.tensor([2]), torch.finfo(o.dtype).min))
    s2, _ = ex.clean_routing(m, probe, "cpu"); hook.remove()
    worst = max(float(np.abs(d_own).max()), float(np.abs(d_fut).max()))
    check("forcing a token onto its own experts changes nothing", worst < 1e-5, f"max |change| {worst:.1e}")
    check("nearest flip changes exactly one expert", one_diff)
    check("a disabled expert is never selected", bool((s2[1] != 2).all()))


def test_min_upstream_flip():
    m, probe, pos = setup_research()
    rows = [r for r in ex.min_upstream_flip(m, probe[:2], pos, "cpu") if r["min_up_dist"] == r["min_up_dist"]]
    cap = ex.Capture(m); good = 0
    for r in rows[:12]:
        t = int(r["pos"]); L = r["layer"]; x = torch.from_numpy(probe[r["window"], :-1])[None]
        with torch.enable_grad():
            cap.emb_noise = None; m(x)
            z = ex.router_logits(m, L, cap.h[L])[0, t]; v = z.topk(3).values
            g = torch.autograd.grad(v[1] - v[2], cap.emb)[0]
        u = -g / g.norm(); base = set(z.topk(2).indices.tolist())

        def top_at(a):
            cap.emb_noise = u * a
            with torch.no_grad():
                m(x); return set(ex.router_logits(m, L, cap.h[L])[0, t].topk(2).indices.tolist())
        good += (top_at(r["min_up_dist"]) != base) and (top_at(0.97 * r["min_up_dist"]) == base)
    cap.close()
    check("minimum upstream flip distance is a real boundary", good == min(12, len(rows)), f"{good}/{min(12, len(rows))}")


def test_ft_causes():
    m, probe, pos = setup_research()
    ft = tiny(); ft.load_state_dict(m.state_dict())
    with torch.no_grad():
        for p in ft.parameters():
            p.add_(0.05 * torch.randn_like(p))
    causes = ex.ft_flip_causes(m, ft, probe, "cpu")
    a, _ = ex.clean_routing(m, probe, "cpu"); b, _ = ex.clean_routing(ft, probe, "cpu")
    flips = (a[1] != b[1]).any(-1)
    check("fine-tuning flip causes agree with the flips themselves", bool(((causes[1] > 0) == flips).all()),
          f"{int(flips.sum())} flips")


def test_boot_auc_ap():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(0)
    n, nw = 600, 30
    groups = np.repeat(np.arange(nw), n // nw)
    y = (rng.random(n) < 0.3).astype(int)
    pred = np.round(rng.random(n) + 0.3 * y, 1)          # rounded -> many tied scores
    W, ref_auc, ref_ap = [], [], []
    for _ in range(40):
        pick = rng.integers(0, nw, nw)
        idx = np.concatenate([np.where(groups == w)[0] for w in pick])
        W.append(np.bincount(pick, minlength=nw)[groups])
        ref_auc.append(roc_auc_score(y[idx], pred[idx])); ref_ap.append(average_precision_score(y[idx], pred[idx]))
    auc, ap = ex.boot_auc_ap(y, pred, np.array(W, dtype=np.float32), chunk=7)
    err = max(np.abs(auc - ref_auc).max(), np.abs(ap - ref_ap).max())
    check("fast bootstrap AUROC/AUPRC equal sklearn on resampled data (with ties)", err < 1e-5, f"max diff {err:.1e}")


def test_holm():
    rng = np.random.default_rng(1)
    scores = {k: (rng.random(50), rng.random(50)) for k in ex.FEATURE_SETS}
    rows = ex.paired_tests(np.array([0, 1] * 30), {k: rng.random(60) for k in ex.FEATURE_SETS}, scores, "y")
    check("Holm-corrected p-values are never smaller than raw ones",
          all(r["p_auroc_holm"] >= r["p_auroc"] for r in rows) and sum(r["primary"] for r in rows) == 1)


# ------------------------------------------------------------------ multi-dataset data and chat (course feedback 2)
class FakeTok:
    """Character-level stand-in for the BPE tokenizer: ids 4.. are characters (0-3 are the special tokens)."""
    sp = {"<|endoftext|>": 0, "<|pad|>": 1, "<|instruction|>": 2, "<|response|>": 3}

    class _E:
        def __init__(self, ids): self.ids = ids

    def encode(self, text): return self._E([4 + (ord(c) % 250) for c in text])
    def decode(self, ids): return "".join(chr(i - 4) for i in ids if i >= 4)
    def token_to_id(self, t): return self.sp[t]


def test_chat_encoding():
    from finetune import encode_conversation
    tok, sp = FakeTok(), FakeTok.sp
    conv = [{"role": "system", "content": "Be kind."}, {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "Joke?"},
            {"role": "assistant", "content": "No."}]
    ids, mask = encode_conversation(tok, conv, sp, block_size=100)
    a1, a2 = tok.encode("Hello!").ids + [0], tok.encode("No.").ids + [0]
    expect = ([2] + tok.encode("Be kind.\n\nHi").ids + [3] + a1 + [2] + tok.encode("Joke?").ids + [3] + a2)
    masked = [t for t, m in zip(ids, mask) if m]
    short = encode_conversation(tok, conv, sp, block_size=len(expect) - 2)      # second turn no longer fits
    none = encode_conversation(tok, conv[:3], sp, block_size=5)                 # not even the first turn fits
    check("chat format: user turns masked out, assistant tokens + <|endoftext|> trained; whole turns cut when too long",
          ids == expect and masked == a1 + a2 and short is not None and short[0] == expect[:len(short[0])]
          and short[0][-1] == 0 and sum(short[1]) == len(a1) and none is None)


def test_chat_history():
    from chat import Chat
    m = tiny(moe=False, block_size=32)
    bot = Chat(m, FakeTok(), max_new_tokens=6)
    bot.turns = [[2, 10, 11, 3, 20, 0], [2, 12, 13, 14, 3, 21, 22, 0], [2, 15, 3, 23, 24, 25, 0]]
    user = [2, 16, 17, 3]
    ctx = bot._context(user)                              # budget 32 - 6 = 26 tokens
    flat = sum(bot.turns, []) + user
    ok_ctx = len(ctx) <= 26 and ctx[0] == 2 and flat[-len(ctx):] == ctx
    big = [[2] + [9] * 10 + [3, 5, 0]] * 5                # 5 turns of 14 tokens: only 1 fits beside the user turn
    bot.turns = big
    ctx2 = bot._context(user)
    bot.reset()
    r = bot.reply("Hello")
    check("chat memory: oldest whole turns are dropped to fit the context; each turn is stored ending in <|endoftext|>",
          ok_ctx and len(ctx2) == 14 + 4 and isinstance(r, str) and len(bot.turns) == 1
          and bot.turns[0][0] == 2 and bot.turns[0][-1] == 0)


def test_data_mix():
    import json, subprocess, sys
    from tokenizers import Tokenizer
    d = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, "prepare_data_mix.py", "--out_dir", d, "--synthetic", "--train_tokens", "4e5",
                        "--val_tokens", "1e5", "--tok_train_docs", "3000", "--vocab_size", "1000"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        check("multi-dataset pipeline runs", False, r.stderr[-300:]); return
    meta = json.load(open(os.path.join(d, "meta.json")))
    tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    val = np.fromfile(os.path.join(d, "val.bin"), dtype=np.uint16)
    src = np.fromfile(os.path.join(d, "val_source.bin"), dtype=np.uint8)
    train = np.fromfile(os.path.join(d, "train.bin"), dtype=np.uint16)
    shares_ok = all(abs(meta["actual_train_shares"][k] - meta["target_shares"][k]) < 0.03 for k in meta["sources"])
    half = len(src) // 2
    halves_ok = all(((src[:half] == i).mean() > 0.1 and (src[half:] == i).mean() > 0.1) for i in range(3))
    import re
    ids = lambda arr: set(re.findall(r"[a-z_]+ doc\d{7}", tok.decode(arr.tolist())))
    leak = ids(val) & ids(train)
    # the source label changes only right after an <|endoftext|>: labels follow whole documents
    change = np.flatnonzero(np.diff(src.astype(int)) != 0)
    aligned = bool((val[change] == 0).all())
    check("data mix: token shares match targets, both validation halves contain every source, "
          "no validation document in training, source labels follow documents",
          shares_ok and halves_ok and not leak and aligned and len(src) == len(val),
          f"shares {meta['actual_train_shares']}, leaked {len(leak)}")
    test_data_mix.dir = d


def test_probe_sources():
    from routing_analysis import get_probe_set, probe_sources
    d = getattr(test_data_mix, "dir", None)
    if d is None:
        check("probe tokens mapped to their data source", False, "data-mix test did not run"); return
    probe = get_probe_set(d, n_seq=6, T=40)
    src, names = probe_sources(d, n_seq=6, T=40)
    val = np.fromfile(os.path.join(d, "val.bin"), dtype=np.uint16)
    allsrc = np.fromfile(os.path.join(d, "val_source.bin"), dtype=np.uint8)
    ok = True
    for w in range(6):                       # locate each probe window in val.bin and compare the labels
        hits = [i for i in range(len(val) - 40) if (val[i:i + 41] == probe[w]).all()]
        ok &= bool(hits) and any((allsrc[i:i + 40] == src[w]).all() for i in hits)
    check("probe tokens are mapped to the right data source", ok and src.shape == (6, 40) and len(names) == 3)



def test_finished_runs_and_half_snapshots():
    import subprocess, sys
    from routing_analysis import load_model
    d = getattr(test_data_mix, "dir", None)
    if d is None:
        check("finished runs skip; float16 snapshots load", False, "data-mix test did not run"); return
    out = tempfile.mkdtemp()
    cmd = [sys.executable, "train.py", "--data_dir", d, "--out_dir", out, "--preset", "tiny_moe8_adv", "--max_steps", "4",
           "--eval_interval", "2", "--eval_iters", "1", "--snapshot_interval", "2", "--batch_size", "2", "--grad_accum", "1",
           "--warmup_steps", "1", "--device", "cpu"]
    r1 = subprocess.run(cmd, capture_output=True, text=True)
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    last = torch.load(os.path.join(out, "ckpt_last.pt"), weights_only=False)
    snap = torch.load(os.path.join(out, "snapshot_step000004.pt"), weights_only=False)
    m32, m16 = load_model(os.path.join(out, "ckpt_last.pt"), "cpu"), load_model(os.path.join(out, "snapshot_step000004.pt"), "cpu")
    x = torch.randint(4, 300, (2, 32))
    diff = (m32(x)[0] - m16(x)[0]).abs().max().item()
    check("finished run keeps weights only and a rerun skips training; float16 snapshots load and match float32",
          r1.returncode == 0 and r2.returncode == 0 and "already finished" in r2.stdout and "optimizer" not in last
          and next(iter(snap["model"].values())).dtype == torch.float16 and diff < 0.05, f"logit diff {diff:.1e}")


def test_drive_friendly_saving():
    """Resumable checkpoints store Adam's moments in bfloat16 without touching the live optimizer, load back as float32
    and continue almost identically; a save that does not fit returns False instead of crashing."""
    import copy
    import train as tr
    torch.manual_seed(0)
    lin = torch.nn.Linear(16, 8)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2)
    for _ in range(3):
        lin(torch.randn(4, 16)).pow(2).sum().backward(); opt.step(); opt.zero_grad()
    full = opt.state_dict()
    small = tr.compact_optimizer_state(full)
    moments = lambda o: [v for st in o.state.values() for k, v in st.items() if k != "step"]
    live_ok = all(v.dtype == torch.float32 for v in moments(opt))
    small_ok = all(v.dtype == torch.bfloat16 for st in small["state"].values() for k, v in st.items() if k != "step")
    lin2 = copy.deepcopy(lin)
    opt2 = torch.optim.AdamW(lin2.parameters(), lr=1e-2)
    opt2.load_state_dict(small)
    back_ok = all(v.dtype == torch.float32 for v in moments(opt2))
    x = torch.randn(4, 16)
    for L, O in [(lin, opt), (lin2, opt2)]:
        L(x).pow(2).sum().backward(); O.step()
    # bfloat16 keeps 8 mantissa bits (about 0.4% relative error), so after one resumed step the weights may differ
    # by a few percent of one learning-rate step; a real bug (lost or zeroed moments) gives a difference near lr itself.
    diff = (lin.weight - lin2.weight).abs().max().item()
    ratio = tr.nbytes(small) / tr.nbytes(full)
    out = tempfile.mkdtemp()
    real = tr.shutil.disk_usage
    tr.shutil.disk_usage = lambda p: real(p)._replace(free=0)
    try:
        refused = tr.safe_save({"w": torch.zeros(10)}, os.path.join(out, "x.pt")) is False
    finally:
        tr.shutil.disk_usage = real
    refused = refused and not os.path.exists(os.path.join(out, "x.pt"))
    saved = tr.safe_save({"w": torch.zeros(10)}, os.path.join(out, "x.pt")) and os.path.exists(os.path.join(out, "x.pt"))
    check("Drive-friendly saving: bfloat16 optimizer moments (live state untouched), float32 after loading, same next "
          "step; a save that does not fit is skipped with a warning, not a crash",
          live_ok and small_ok and back_ok and diff < 0.1 * 1e-2 and ratio < 0.6 and refused and saved,
          f"size ratio {ratio:.2f}, weight diff after one step {diff:.1e}")


def test_run_lock():
    """Two sessions on one shared run folder: a fresh heartbeat blocks a second copy; a stale one does not;
    a finished run leaves DONE.json and removes its heartbeat."""
    import json, subprocess, sys, time
    d = getattr(test_data_mix, "dir", None)
    if d is None:
        check("run lock: one session per run folder", False, "data-mix test did not run"); return
    out = tempfile.mkdtemp()
    lock = os.path.join(out, "RUNNING")
    cmd = [sys.executable, "train.py", "--data_dir", d, "--out_dir", out, "--preset", "tiny_adv", "--max_steps", "2",
           "--eval_interval", "2", "--eval_iters", "1", "--batch_size", "2", "--grad_accum", "1", "--warmup_steps", "1",
           "--device", "cpu"]
    json.dump({"t": time.time() - 30, "id": "other"}, open(lock, "w"))           # someone else is training
    r1 = subprocess.run(cmd, capture_output=True, text=True)
    blocked = "Another session" in r1.stdout and not os.path.exists(os.path.join(out, "log.csv"))
    json.dump({"t": time.time() - 3600, "id": "other"}, open(lock, "w"))         # that session died an hour ago
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    done = json.load(open(os.path.join(out, "DONE.json"))) if os.path.exists(os.path.join(out, "DONE.json")) else {}
    os.remove(os.path.join(out, "DONE.json"))
    r3 = subprocess.run(cmd, capture_output=True, text=True)                    # finished run: marker rewritten
    check("run lock: a live session blocks a second copy, a stale heartbeat does not; finished runs leave DONE.json",
          r1.returncode == 0 and blocked and r2.returncode == 0 and done.get("step") == 2 and not os.path.exists(lock)
          and "already finished" in r3.stdout and os.path.exists(os.path.join(out, "DONE.json")),
          f"blocked={blocked}, done={done}")


# ------------------------------------------------------------------ public MoE (Hugging Face Granite) adapter
def _tiny_granite():
    from transformers import GraniteMoeConfig, GraniteMoeForCausalLM
    cfg = GraniteMoeConfig(vocab_size=300, hidden_size=64, intermediate_size=32, num_hidden_layers=3,
                           num_attention_heads=4, num_key_value_heads=2, num_local_experts=6, num_experts_per_tok=2,
                           max_position_embeddings=128, embedding_multiplier=12.0, logits_scaling=6.0,
                           residual_multiplier=0.22, attention_multiplier=0.015625)
    torch.manual_seed(0)
    return GraniteMoeForCausalLM._from_config(cfg, attn_implementation="eager", experts_implementation="eager").eval()


def test_hf_adapter():
    try:
        hf = _tiny_granite()
    except Exception as e:
        print(f"[SKIP] public-MoE adapter tests (transformers not available: {type(e).__name__})")
        return
    from hf_moe import HFMoE
    x = torch.randint(0, 300, (2, 32))
    ref = hf(input_ids=x).logits.detach()
    m = HFMoE(hf, layers=(0, 2), block_size=32).eval()
    out = m(x, return_routing=True)[0].detach()
    own = m.blocks[1].ffn.last_routing["top_idx"]
    m.blocks[1].ffn.force_route = (torch.ones(64, dtype=torch.bool), own)
    forced = m(x)[0].detach()
    m.blocks[1].ffn.force_route = None
    other = own.clone(); other[5] = torch.tensor([o for o in range(6) if o not in own[5].tolist()][:2])
    m.blocks[1].ffn.force_route = (torch.ones(64, dtype=torch.bool), other)
    changed = m(x)[0].detach()
    m.blocks[1].ffn.force_route = None
    shared = m.blocks[0].ffn.router.weight is hf.model.layers[0].block_sparse_moe.router.weight
    check("public MoE adapter: identical logits to the original Granite model; router weight shared; "
          "forcing a token onto its own experts changes nothing, onto others changes it",
          torch.allclose(out, ref, atol=1e-5) and torch.allclose(forced, out, atol=1e-6)
          and not torch.allclose(changed, out) and shared and m.cfg.top_k == 2 and m.cfg.n_experts == 6,
          f"max diff {(out - ref).abs().max():.1e}")

    # geometry on the public-model wrapper, against direct autograd computations (layer 2)
    probe = np.random.default_rng(1).integers(4, 300, (1, 33)).astype(np.int64)
    pos = ex.subsample_positions(32, 6, start=6)
    rows = ex.geometry_features(m, probe, pos, "cpu", n_power=300, n_hutch=3000, pos_chunk=3)
    r = [z for z in rows if z["layer"] == 2][2]
    t = int(pos[2]); w = torch.from_numpy(probe[0]); xx = w[:-1][None]; yy = w[1:][None]
    cap = ex.Capture(m)
    with ex.math_attention():
        def margin_at(delta):
            cap.emb_noise = delta
            m(xx)
            v = m.blocks[2].ffn.router(cap.h[2])[0, t].float().topk(3).values
            return v[1] - v[2]
        m(xx); d = torch.zeros_like(cap.emb.detach(), requires_grad=True)
        g = torch.autograd.grad(margin_at(d), d, create_graph=True)[0]
        u = (-g / g.norm()).detach()
        c_ref = (torch.autograd.grad((g * u).sum(), d)[0] * u).sum().item()
        cap.emb_noise = None
        m(xx); h0 = cap.h[2].detach().clone()

        def Lt_of(ht):
            def hook(mod, inp):
                hh = inp[0].clone(); hh[0, t] = ht; return (hh,)
            hd = m.blocks[2].ffn.register_forward_pre_hook(hook)
            lg = m(xx)[0]; hd.remove()
            return F.cross_entropy(lg[0, t].float(), yy[0, t])
        H = torch.autograd.functional.hessian(Lt_of, h0[0, t])
    cap.close()
    ev = torch.linalg.eigvalsh(H); top = ev[ev.abs().argmax()].item()
    off = (H ** 2).sum() - (torch.diagonal(H) ** 2).sum(); se = math.sqrt(2 * off.item() / 3000)
    check("public MoE adapter: margin gradient, margin curvature, Hessian top eigenvalue and trace match direct autograd",
          abs(r["margin_sens"] - g.norm().item()) < 1e-4 * g.norm().item()
          and abs(r["margin_curv"] - c_ref) < 1e-3 * abs(c_ref) + 1e-5
          and abs(r["hess_top"] - top) < 0.02 * abs(top) + 1e-7
          and abs(r["hess_trace"] - H.trace().item()) < 4 * se + 1e-9,
          f"curv {r['margin_curv']:.4f}/{c_ref:.4f}, top {r['hess_top']:.2e}/{top:.2e}")


def test_hf_loading_and_data():
    try:
        hf = _tiny_granite()
        from transformers import PreTrainedTokenizerFast
    except Exception as e:
        print(f"[SKIP] public-MoE loading tests (transformers not available: {type(e).__name__})")
        return
    from tokenizers import Tokenizer
    from routing_analysis import load_model, get_probe_set, probe_sources
    from hf_moe import make_data
    d_src = getattr(test_data_mix, "dir", None)
    if d_src is None:
        check("public MoE: load + re-tokenized probe data", False, "data-mix test did not run"); return
    mdir = tempfile.mkdtemp()
    hf.save_pretrained(mdir)
    tk = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(os.path.join(d_src, "tokenizer.json")),
                                 eos_token="<|endoftext|>", pad_token="<|pad|>")
    tk.save_pretrained(mdir)
    m = load_model(f"hf:{mdir}?layers=0,2&T=32", "cpu")
    x = torch.randint(0, 300, (2, 16))
    same = torch.allclose(m(x)[0], hf(input_ids=x).logits, atol=1e-5)
    out = tempfile.mkdtemp()
    meta = make_data(d_src, out, mdir, max_docs=200)
    ours = Tokenizer.from_file(os.path.join(d_src, "tokenizer.json"))
    val = np.fromfile(os.path.join(out, "val.bin"), dtype=np.uint16)
    eos = meta["special_tokens"]["<|endoftext|>"]
    first_doc = ours.decode(val[: int(np.argmax(val == eos))].tolist())
    src_val = np.fromfile(os.path.join(d_src, "val.bin"), dtype=np.uint16)
    first_src = ours.decode(src_val[: int(np.argmax(src_val == 0))].tolist())
    probe = get_probe_set(out, n_seq=4, T=24)
    src, names = probe_sources(out, n_seq=4, T=24)
    check("public MoE: loads with the 'hf:' prefix (identical logits); probe data re-tokenized document by document "
          "with sources kept", same and m.cfg.moe_layers == (0, 2) and first_doc == first_src
          and probe.shape == (4, 25) and src.shape == (4, 24) and len(val) == len(np.fromfile(os.path.join(out, "val_source.bin"), dtype=np.uint8)))



def test_quantization_and_repair():
    from dataclasses import asdict
    m = tiny(**ADV)
    W0 = {k: v.clone() for k, v in m.state_dict().items()}
    n = ex.fake_quantize_(m, bits=4, group=16)
    sd = m.state_dict()
    untouched = torch.equal(sd["tok_emb.weight"], W0["tok_emb.weight"]) and torch.equal(sd["blocks.1.ffn.router.weight"], W0["blocks.1.ffn.router.weight"])
    w0, w1 = W0["blocks.0.ffn.up.weight"].reshape(-1, 16), sd["blocks.0.ffn.up.weight"].reshape(-1, 16)
    bound = bool(((w0 - w1).abs() <= w0.abs().amax(1, keepdim=True) / 7 / 2 + 1e-7).all())
    levels = len(torch.unique((w1 / (w0.abs().amax(1, keepdim=True) / 7)).round())) <= 16
    snap = {k: v.clone() for k, v in m.state_dict().items()}
    ex.fake_quantize_(m, bits=4, group=16)
    idem = all(torch.allclose(snap[k], v, atol=1e-6) for k, v in m.state_dict().items())
    check("4-bit group quantization: error within half a step, at most 16 levels, idempotent; embeddings and routers untouched",
          untouched and bound and levels and idem and n > 0)

    # full Phase 8 on a random model and the multi-dataset synthetic data
    d = getattr(test_data_mix, "dir", None)
    if d is None:
        check("quantization phase", False, "data-mix test did not run"); return
    V = json.load(open(os.path.join(d, "meta.json")))["vocab_size"]
    cfg = ModelConfig(vocab_size=V, block_size=32, n_layer=2, n_head=4, n_kv_head=2, qk_norm=True, n_embd=32,
                      ffn_hidden=64, expert_hidden=32, n_experts=4, top_k=2, moe_layers=(1,))
    torch.manual_seed(3)
    mm = NovaLM(cfg)
    ck = os.path.join(tempfile.mkdtemp(), "m.pt")
    torch.save({"model": mm.state_dict(), "config": asdict(cfg), "step": 0, "best_val": 0.0}, ck)
    out = tempfile.mkdtemp()
    df = ex.build_table(ck, None, d, out, "cpu", n_seq=6, n_pos=8, n_samples=2, n_power=3, n_hutch=2,
                        sigmas=(0.1, 0.4), quant_bits=2, quant_group=32)
    flip = df.quant_flip == 1
    zero_ok = bool((df.loc[~flip, "quant_route_damage"].abs() < 1e-5).all())
    nonzero = bool((df.loc[flip, "quant_route_damage"].abs() > 1e-6).any()) if flip.any() else False
    # repair curve on synthetic damage
    rng = np.random.default_rng(0)
    toy = pd.DataFrame({"window": np.repeat(np.arange(20), 10), "dmg": rng.normal(0, 1, 200)})
    good = toy.dmg.values + rng.normal(0, 0.3, 200)
    rows = pd.DataFrame(ex.repair_curve(toy, {"good": good, "noise": rng.random(200)}, "dmg", n_boot=50))
    at = lambda s, b: rows[(rows.selector == s) & (rows.budget == b)].recovered_share.item()
    curve_ok = (at("oracle (true damage)", 0.1) >= at("good", 0.1) > at("noise", 0.1) - 0.05
                and at("random (expected)", 0.2) == 0.2 and at("good", 0.5) >= at("good", 0.1))
    check("quantization phase: route damage is exactly 0 for tokens whose routing did not flip and non-zero for "
          "flipped ones; repair curve: oracle >= good predictor > noise, random = budget",
          zero_ok and nonzero and curve_ok and "quant_token_dloss" in df,
          f"{int(flip.sum())} flips of {len(df)}")



def test_benchmark_scoring():
    import bench_eval as be
    from tokenizers import Tokenizer
    d = getattr(test_data_mix, "dir", None)
    if d is None:
        check("benchmark scoring", False, "data-mix test did not run"); return
    tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    V = tok.get_vocab_size()
    m = NovaLM(ModelConfig(vocab_size=V, block_size=96, n_layer=2, n_head=2, n_embd=32, ffn_hidden=64)).eval()
    enc = be.make_encoder(tok)
    ctx, chs = "Question: what do plants need?\nAnswer:", [" light", " a car that is red"]
    lp = be.choice_logprobs(m, enc, ctx, chs, "cpu")
    ok = True
    for ch, val in zip(chs, lp):                            # manual: one sequence at a time
        ids = enc(ctx) + enc(ch); n = len(enc(ch))
        logp = F.log_softmax(m(torch.tensor([ids[:-1]]))[0][0].float(), -1)
        ref = sum(logp[len(ids) - n - 1 + j, ids[len(ids) - n + j]].item() for j in range(n))
        ok &= abs(ref - val) < 1e-4
    long_ctx = "word " * 400                                 # longer than the window: cut from the left
    lp2 = be.choice_logprobs(m, enc, long_ctx, chs, "cpu")
    hs = be.fmt_hellaswag({"ctx_a": "A man sits.", "ctx_b": "he [title] reads", "activity_label": "Reading",
                           "endings": ["opens a book.", "eats a car."], "label": "0"})
    arc = be.fmt_arc({"question": "Q?", "choices": {"text": ["a", "b", "c"], "label": ["A", "B", "C"]}, "answerKey": "C"})
    sq = be.fmt_sciq({"support": "S", "question": "Q?", "distractor1": "x", "distractor2": "y", "distractor3": "z",
                      "correct_answer": "w"})
    r = be.evaluate_task(m, enc, [(ctx, chs, 0)] * 4, "cpu")
    check("standard benchmarks: choice log-probabilities equal a manual computation, long inputs are cut to the "
          "window, task formats and gold answers follow lm-evaluation-harness",
          ok and len(lp2) == 2 and all(np.isfinite(lp2)) and hs[0] == "Reading: A man sits. He. reads" and hs[2] == 0
          and arc[2] == 2 and arc[1][0] == " a" and sq[2] == 3 and sq[1][3] == " w" and r["n"] == 4)


def test_margin_trace_and_closed_form():
    # (1) Hutchinson estimates of tr(H) and |H|_F^2 for the margin Hessian w.r.t. the embeddings = exact Hessian
    m, probe, _ = setup_research()
    t = 20
    rows = ex.geometry_features(m, probe[:1], [t], "cpu", n_power=3, n_hutch=2, n_hutch_margin=1500)
    r = [x for x in rows if x["layer"] == 1][0]
    w = torch.from_numpy(probe[0]); x = w[:-1][: t + 1][None]
    cap = ex.Capture(m)
    with ex.math_attention():
        m(x); shape = cap.emb.shape
        def margin_flat(d):
            cap.emb_noise = d.view(shape)
            m(x)
            v = m.blocks[1].ffn.router(cap.h[1])[0, t].topk(3).values
            return v[1] - v[2]
        H = torch.autograd.functional.hessian(margin_flat, torch.zeros(shape.numel()))
    cap.emb_noise = None; cap.close()
    tr, fro2 = H.trace().item(), (H ** 2).sum().item()
    se_tr = math.sqrt(2 * fro2 / 1500)
    se_f = math.sqrt(2 * torch.trace(torch.linalg.matrix_power(H, 4)).item() / 1500)
    ok1 = abs(r["margin_hess_trace"] - tr) < 4 * se_tr + 1e-9 and abs(r["margin_hess_fro2"] - fro2) < 4 * se_f + 1e-9
    # (2) the formula's mean and variance are the exact moments of the quadratic model; p1 is exact when H = 0
    rng = np.random.default_rng(0)
    d, sigma, mm = 6, 0.3, 0.4
    g = rng.normal(size=d); A = rng.normal(size=(d, d)); Hq = 0.6 * (A + A.T) / 2
    eps = rng.normal(scale=sigma, size=(400_000, d))
    q = mm + eps @ g + 0.5 * np.einsum("ni,ij,nj->n", eps, Hq, eps)
    lin = mm + eps @ g
    p1, p2 = ex.flip_prob_formula([mm], [np.linalg.norm(g)], [np.trace(Hq)], [(Hq ** 2).sum()], sigma)
    mean_f = mm + 0.5 * sigma ** 2 * np.trace(Hq)
    var_f = sigma ** 2 * (g ** 2).sum() + 0.5 * sigma ** 4 * (Hq ** 2).sum()
    ok2 = (abs(q.mean() - mean_f) < 0.005 and abs(q.var() / var_f - 1) < 0.01 and abs(p1[0] - (lin < 0).mean()) < 0.003
           and abs(p2[0] - (q < 0).mean()) <= abs(p1[0] - (q < 0).mean()) + 0.002)
    check("closed-form flip probability: margin-Hessian trace and Frobenius norm match the exact Hessian; "
          "the formula's mean/variance equal the exact moments; first order exact for a linear margin",
          ok1 and ok2, f"trace {r['margin_hess_trace']:.4f} vs {tr:.4f} (s.e. {se_tr:.4f}); "
                       f"fro2 {r['margin_hess_fro2']:.4f} vs {fro2:.4f}; P(flip) MC {(q < 0).mean():.4f}, p1 {p1[0]:.4f}, p2 {p2[0]:.4f}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    raise SystemExit(0 if all(RESULTS) else 1)
