"""
Team NOVA - AFML Project
train.py: pretraining (dense or MoE) on the FineWeb-Edu/C4/Wikipedia mix (the earlier small model used TinyStories), with resumable checkpoints.

Fixes compared with the video training loop:
  * the loop counts OPTIMIZER steps; gradient accumulation happens inside each step
  * the LR schedule is computed per optimizer step: linear warmup -> cosine decay to min_lr
  * min_lr < lr (the video had min_lr = 5x lr, so its LR went UP)
  * backward() outside the autocast block; GradScaler only for float16
  * weight decay only on matrices (not on norm gains)
  * checkpoints + CSV log saved to Google Drive, and training resumes after a disconnect
  * periodic weight snapshots, kept for the training-dynamics analysis (Phase 7)
  * Drive-friendly: the resumable checkpoint is written every --resume_every_min minutes (old copies of overwritten
    files can sit in Drive's Bin and count against the quota), free space is checked before every save, and a full
    Drive gives a warning instead of a crash
  * one session per run folder (RUNNING heartbeat), and DONE.json when a run has finished

Example (Colab, T4 GPU):
    python train.py --data_dir /content/drive/MyDrive/nova/data \
                    --out_dir  /content/drive/MyDrive/nova/runs/dense --preset dense
"""

import argparse
import atexit
import csv
import json
import math
import os
import shutil
import socket
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch

from model import ModelConfig, NovaLM

# "Base" model for the C4 corpus (course feedback: scale the SLM up), in the GPT-2-small size class:
# 12 blocks, width 768, 12 query heads sharing 4 key/value heads (grouped-query attention), QK-norm.
# Dense ~88M parameters. MoE: 8 experts of half width in blocks 3, 7, 11 (code indices 2, 6, 10),
# ~131M total / ~88M active per token. Exact counts are printed at the start of training and checked in tests.
BASE = dict(block_size=512, n_layer=12, n_head=12, n_kv_head=4, qk_norm=True,
            n_embd=768, ffn_hidden=2048, expert_hidden=1024)
# Fallback if the T4 is too slow for the Base model: the earlier 60M configuration (10 x 640, plain attention).
BASE60 = dict(block_size=512, n_layer=10, n_head=10, n_embd=640, ffn_hidden=1728, expert_hidden=864)

PRESETS = {
    # Phase 1: dense baseline
    "dense": dict(moe_layers=()),
    # Phase 2: MoE in alternate layers, 8 experts, top-2, with load balancing
    "moe": dict(moe_layers=(1, 3, 5)),
    # Phase 2 ablation: same MoE but trained WITHOUT the load-balancing loss
    "moe_nobal": dict(moe_layers=(1, 3, 5)),
    # Base models on C4 (same recipe as above, larger)
    "base_dense": dict(BASE, moe_layers=()),
    "base_moe": dict(BASE, moe_layers=(2, 6, 10)),
    "base_moe_nobal": dict(BASE, moe_layers=(2, 6, 10)),
    # Placement ablation (TA question): MoE in blocks 1, 2, 5 (code indices 0, 1, 4) instead of 3, 7, 11.
    # Same number of MoE layers, so exactly the same parameter count; only WHERE the experts sit changes.
    "base_moe_early": dict(BASE, moe_layers=(0, 1, 4)),
    "moe_early": dict(moe_layers=(0, 1, 4)),
    "base60_dense": dict(BASE60, moe_layers=()),
    "base60_moe": dict(BASE60, moe_layers=(2, 5, 8)),
    "base60_moe_nobal": dict(BASE60, moe_layers=(2, 5, 8)),
    # tiny config for a quick smoke test (runs on CPU in seconds)
    "tiny": dict(n_layer=2, n_head=2, n_embd=64, ffn_hidden=128, expert_hidden=64, block_size=64),
    "tiny_moe": dict(n_layer=2, n_head=2, n_embd=64, ffn_hidden=128, expert_hidden=64,
                     block_size=64, moe_layers=(1,), n_experts=4),
    "tiny_moe8": dict(n_layer=4, n_head=2, n_embd=64, ffn_hidden=128, expert_hidden=64,
                      block_size=64, moe_layers=(1, 3), n_experts=8),
    # tiny stand-ins for the Base model's attention (grouped-query + QK-norm), used by the Base smoke tests
    "tiny_adv": dict(n_layer=2, n_head=4, n_kv_head=2, qk_norm=True, n_embd=64, ffn_hidden=128,
                     expert_hidden=64, block_size=64),
    "tiny_moe8_adv": dict(n_layer=4, n_head=4, n_kv_head=2, qk_norm=True, n_embd=64, ffn_hidden=128,
                          expert_hidden=64, block_size=64, moe_layers=(1, 3), n_experts=8),
}


# Training defaults per preset (anything passed on the command line still wins). The three Base runs
# share one recipe so that dense vs MoE vs no-balancing stay a fair comparison:
# 8 x 16 x 512 = 65,536 tokens per step x 6,000 steps = 393M tokens. Micro-batch 8 keeps the 88M model
# well inside the T4's 15 GB; the effective batch is 8 x 16 = 128 sequences, the same as before.
PRESET_TRAIN = {
    name: dict(batch_size=8, grad_accum=16, max_steps=6000, lr=6e-4, min_lr=6e-5, snapshot_interval=1500)
    for name in ("base_dense", "base_moe", "base_moe_nobal", "base_moe_early",
                 "base60_dense", "base60_moe", "base60_moe_nobal")
}


def uses_balancing(preset):
    """The *_nobal presets are the ablation: same MoE, trained WITHOUT the load-balancing loss."""
    return not preset.endswith("nobal")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--preset", default="dense", choices=list(PRESETS))
    p.add_argument("--max_steps", type=int, default=10000, help="optimizer steps")
    p.add_argument("--batch_size", type=int, default=32, help="sequences per micro-batch")
    p.add_argument("--grad_accum", type=int, default=4, help="micro-batches per optimizer step")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--aux_coef", type=float, default=0.01, help="load-balancing loss weight")
    p.add_argument("--z_coef", type=float, default=1e-3, help="router z-loss weight")
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--snapshot_interval", type=int, default=1000,
                   help="save weights-only snapshots for the training-dynamics analysis")
    p.add_argument("--resume_every_min", type=float, default=45,
                   help="write the resumable checkpoint (weights + optimizer) to out_dir at most this often")
    p.add_argument("--compact_optimizer", action="store_true",
                   help="store Adam's moments in bfloat16 in the resumable checkpoint (smaller files on Google Drive)")
    p.add_argument("--init_from", default="",
                   help="continue a run from a weight snapshot (snapshot_stepXXXXXX.pt): weights + step from the "
                        "snapshot, fresh optimizer state, log rows after that step removed (recovery after a broken run)")
    p.add_argument("--save_best", action="store_true",
                   help="also keep ckpt_best.pt (not used later: pretrained models are always taken at the last step)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    preset = p.parse_known_args()[0].preset
    p.set_defaults(**PRESET_TRAIN.get(preset, {}))
    return p.parse_args()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def val_dev_end(val_path, eot_id=0):
    """Index where the validation file is split in two (validation / test), at a document boundary.
    [0, dev_end)  = development half: research probe set, forgetting checks during fine-tuning
    [dev_end, :)  = TEST half: touched only by final_eval.py (notebook 06)."""
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    mid = len(data) // 2
    return mid + int(np.argmax(data[mid:] == eot_id)) + 1


class Batcher:
    """Random windows of block_size+1 tokens from a uint16 memmap: x = w[:-1], y = w[1:].
    limit: only sample windows that end before this token index (e.g. the development half)."""

    def __init__(self, path, block_size, batch_size, device, seed, limit=None):
        self.path, self.T, self.B, self.device = path, block_size, batch_size, device
        self.n = len(np.memmap(path, dtype=np.uint16, mode="r"))
        if limit is not None:
            self.n = min(self.n, int(limit))
        self.rng = np.random.default_rng(seed)

    def get(self):
        data = np.memmap(self.path, dtype=np.uint16, mode="r")   # re-open: avoids memmap RAM growth
        ix = self.rng.integers(0, self.n - self.T - 1, size=self.B)
        w = np.stack([data[i : i + self.T + 1] for i in ix]).astype(np.int64)
        w = torch.from_numpy(w)
        x, y = w[:, :-1].contiguous(), w[:, 1:].contiguous()
        if self.device.startswith("cuda"):
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True))
        return x.to(self.device), y.to(self.device)


def lr_at(step, args):
    """Linear warmup to lr, then cosine decay to min_lr at max_steps."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def make_optimizer(model, args):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    fused = args.device.startswith("cuda")
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=fused)


@torch.no_grad()
def evaluate(model, batchers, eval_iters, ctx):
    model.eval()
    out = {}
    for split, b in batchers.items():
        b.rng = np.random.default_rng(999)       # same eval batches every time -> comparable numbers
        losses = []
        for _ in range(eval_iters):
            x, y = b.get()
            with ctx:
                _, loss, _ = model(x, y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


def sample_text(model, tok, device, eot_id, prompt="Once upon a time", n_tokens=120):
    if tok is None:
        return ""
    idx = torch.tensor([tok.encode(prompt).ids], device=device)
    out = model.generate(idx, n_tokens, temperature=0.8, top_k=50, eot_id=eot_id)
    model.train()
    return tok.decode(out[0].tolist())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
LOCK_STALE_S = 600        # a RUNNING heartbeat older than 10 minutes belongs to a dead session


def read_lock(path):
    """(seconds since the last heartbeat, session id) of a RUNNING file, or (None, None).
    A heartbeat written on THIS machine by a process that no longer exists (e.g. after a kernel restart) counts as
    stale at once; one written on another machine counts as alive until it is LOCK_STALE_S old."""
    try:
        with open(path) as f:
            d = json.load(f)
        age = time.time() - float(d["t"])
    except Exception:
        return None, None
    if os.name != "nt" and d.get("host") == socket.gethostname() and d.get("pid"):   # (on Windows os.kill(pid, 0)
                                                                                     #  would send Ctrl+C, so skip)
        try:
            os.kill(int(d["pid"]), 0)          # signal 0: only checks that the process exists
        except ProcessLookupError:
            age = float("inf")
        except (PermissionError, OSError, ValueError):
            pass
    return age, d.get("id")


def write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


# ---------------- saving to Google Drive without filling it ----------------
# Every overwrite of a file on Drive can leave the old copy in Drive's Bin, which counts against the 15 GB quota.
# So big files are written rarely: the resumable checkpoint every --resume_every_min minutes (not at every evaluation),
# with Adam's two moment estimates stored in bfloat16 (one third smaller), and every save checks the free space first.
def compact_optimizer_state(sd):
    """Copy of optimizer.state_dict() with floating-point state (Adam's moments) in bfloat16. The live optimizer is not
    touched. load_state_dict casts the values back to the parameters' float32; the rounding happens once per resume."""
    small = lambda k, v: v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() and k != "step" else v
    return {"state": {i: {k: small(k, v) for k, v in st.items()} for i, st in sd["state"].items()},
            "param_groups": sd["param_groups"]}


def nbytes(obj):
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(nbytes(v) for v in obj)
    return 0


DRIVE_FULL = ("Google Drive may be full: open drive.google.com -> Bin -> Empty bin (and delete files you do not need). "
              "Training continues and tries again at the next evaluation.")


def safe_save(state, path):
    """Atomic save (write .tmp, then rename). Returns False (with a clear message) instead of crashing when there is
    not enough free space, so a full Drive never kills a training run."""
    try:
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free
    except OSError:
        free = None
    need = 1.1 * nbytes(state) + 50e6
    if free is not None and free < need:
        print(f"WARNING: only {free / 1e9:.2f} GB free, {os.path.basename(path)} needs {need / 1e9:.2f} GB; not saved. "
              + DRIVE_FULL)
        return False
    try:
        with open(path + ".tmp", "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())          # on disk before the rename: a power cut cannot leave a half-written file
        os.replace(path + ".tmp", path)
        return True
    except (OSError, RuntimeError) as e:
        try:
            os.remove(path + ".tmp")
        except OSError:
            pass
        print(f"WARNING: could not save {os.path.basename(path)} ({type(e).__name__}: {str(e)[:120]}). " + DRIVE_FULL)
        return False


def save_until_done(state, path, tries=60):
    """The final weights must reach Drive: retry every minute (up to an hour) while the user frees space."""
    for i in range(tries):
        if safe_save(state, path):
            return True
        print(f"   retrying in 60 s ({i + 1}/{tries}); the trained model is still in memory, do not stop the cell")
        time.sleep(60)
    raise RuntimeError(f"could not save {path}")


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    eot_id = meta["special_tokens"]["<|endoftext|>"]
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    except Exception:
        tok = None

    cfg = ModelConfig(vocab_size=meta["vocab_size"], **PRESETS[args.preset])
    use_aux = uses_balancing(args.preset)
    device = args.device
    model = NovaLM(cfg).to(device)
    optimizer = make_optimizer(model, args)

    # mixed precision: bfloat16 if the GPU supports it (A100/L4), else float16 + GradScaler (T4)
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        dtype, ctx = torch.float32, nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    # ---------------- one session per run folder ----------------
    # Teammates share one Drive folder. Two Colab sessions training the same run would overwrite each other's
    # checkpoints, so a running session keeps a heartbeat in RUNNING (rewritten every 2 minutes) and a second
    # session refuses to start while that heartbeat is fresh (stale after 10 min, or at once if its process is gone).
    lock_path = os.path.join(args.out_dir, "RUNNING")
    done_path = os.path.join(args.out_dir, "DONE.json")
    session = uuid.uuid4().hex
    if not args.no_resume:
        age, owner = read_lock(lock_path)
        if age is not None and age < LOCK_STALE_S and owner != session:
            print(f"Another session is training {args.out_dir} right now (heartbeat {age / 60:.1f} min ago); "
                  f"not starting a second copy. If that session is dead, run this again in "
                  f"{(LOCK_STALE_S - age) / 60:.0f} min, or delete {lock_path}.")
            return

    # ---------------- resume ----------------
    ckpt_path = os.path.join(args.out_dir, "ckpt_last.pt")
    step, best_val = 0, float("inf")
    rewarm_from = None                    # set when continuing from a snapshot (fresh Adam moments need a short warmup)
    def all_finite(sd):
        return all(torch.isfinite(v).all().item() for v in sd.values() if torch.is_tensor(v) and v.is_floating_point())

    log_path = os.path.join(args.out_dir, "log.csv")
    if args.init_from:
        # recovery: weights + step from a snapshot, fresh optimizer; the LR schedule continues from that step
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        if not all_finite(ck["model"]):
            raise SystemExit(f"{args.init_from} contains NaN/inf weights; choose an earlier snapshot.")
        model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["model"].items()})
        step = int(ck["step"])
        if os.path.exists(log_path):                      # keep only the log rows up to the snapshot
            rows = list(csv.reader(open(log_path)))
            keep = [rows[0]] + [r for r in rows[1:] if r and int(r[0]) <= step]
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerows(keep)
            vals = [float(r[4]) for r in keep[1:] if math.isfinite(float(r[4]))]
            best_val = min(vals) if vals else float("inf")
        for f in (ckpt_path, done_path):
            if os.path.exists(f):
                os.remove(f)
        rewarm_from = step
        print(f"Continuing from snapshot step {step} ({args.init_from}) with a fresh optimizer state "
              f"(learning rate re-warmed over 100 steps)")
    elif os.path.exists(ckpt_path) and not args.no_resume:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if not all_finite(ck["model"]):
            raise SystemExit(f"{ckpt_path} contains NaN/inf weights (broken run). Continue from the last good "
                             f"snapshot instead: add --init_from {args.out_dir}/snapshot_stepXXXXXX.pt")
        if "optimizer" not in ck:                         # finished runs keep weights only (saves Drive space)
            print(f"Run already finished at step {ck['step']} (best val {ck['best_val']:.4f}); nothing to do.")
            if not os.path.exists(done_path):
                write_json(done_path, {"step": ck["step"], "best_val": ck["best_val"]})
            return
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        # Make sure Adam's moments are float32 like the weights. (Older checkpoints stored them in bfloat16; on the GPU
        # the fused AdamW kernel must not receive bfloat16 moments for float32 weights.)
        for st in optimizer.state.values():
            for k, v in list(st.items()):
                if k != "step" and torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.float32:
                    st[k] = v.float()
        scaler.load_state_dict(ck["scaler"])
        step, best_val = ck["step"], ck["best_val"]
        if os.path.exists(log_path):                      # drop log rows written after this checkpoint
            rows = list(csv.reader(open(log_path)))
            keep = [rows[0]] + [r for r in rows[1:] if r and int(r[0]) <= step]
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerows(keep)
        print(f"Resumed from step {step} (best val {best_val:.4f})")

    batchers = {
        "train": Batcher(os.path.join(args.data_dir, "train.bin"), cfg.block_size, args.batch_size, device, args.seed + step),
        "val": Batcher(os.path.join(args.data_dir, "val.bin"), cfg.block_size, args.batch_size, device, 10_000 + step),
    }
    eval_batchers = {k: Batcher(b.path, cfg.block_size, args.batch_size, device, 999) for k, b in batchers.items()}

    tokens_per_step = args.batch_size * args.grad_accum * cfg.block_size
    print(f"Preset: {args.preset} | params: {model.num_params()/1e6:.2f}M total, "
          f"{model.num_active_params()/1e6:.2f}M active per token")
    print(f"Device: {device} | dtype: {dtype} | tokens/step: {tokens_per_step:,} | "
          f"total tokens: {tokens_per_step * args.max_steps / 1e6:.0f}M")
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({"model": asdict(cfg), "train": vars(args)}, f, indent=2)

    new_log = not os.path.exists(log_path) or step == 0
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "tokens", "lr", "train_loss", "val_loss", "aux_loss", "z_loss", "tok_per_s"])

    def state(full=False, half=False):
        """full: also optimizer + scaler (needed to resume). half: weights in float16 (snapshots: half the size;
        only used for routing analysis, where float16 rounding is far below the margins we measure)."""
        sd = model.state_dict()
        if half:
            sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
        st = {"model": sd, "config": asdict(cfg), "step": step, "best_val": best_val}
        if full:
            opt_sd = optimizer.state_dict()
            st.update(optimizer=compact_optimizer_state(opt_sd) if args.compact_optimizer else opt_sd,
                      scaler=scaler.state_dict())
        return st

    def save(path, full=True, half=False):       # atomic: a disconnect never leaves a broken file
        return safe_save(state(full, half), path)

    def beat():
        if not args.no_resume:
            write_json(lock_path, {"t": time.time(), "id": session, "host": socket.gethostname(), "pid": os.getpid()})

    def release():                       # on normal exit, an error or a manual stop (not when Colab kills the VM)
        if not args.no_resume and read_lock(lock_path)[1] == session:
            try:
                os.remove(lock_path)
            except OSError:
                pass

    beat()
    atexit.register(release)
    last_beat = last_full = time.time()

    model.train()
    t0, last_aux, last_z = time.time(), float("nan"), float("nan")
    bad_steps = 0
    while step < args.max_steps:
        if time.time() - last_beat > 120:
            beat()
            last_beat = time.time()
        lr = lr_at(step, args)
        if rewarm_from is not None and step - rewarm_from < 100:
            lr *= (step - rewarm_from + 1) / 100
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ---- one optimizer step = grad_accum micro-batches ----
        for _ in range(args.grad_accum):
            x, y = batchers["train"].get()
            with ctx:
                _, loss, info = model(x, y)
                total = loss
                if info["aux_loss"] is not None:
                    if use_aux:
                        total = total + args.aux_coef * info["aux_loss"]
                    total = total + args.z_coef * info["z_loss"]
                    last_aux, last_z = info["aux_loss"].item(), info["z_loss"].item()
            scaler.scale(total / args.grad_accum).backward()      # backward outside autocast
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if dtype != torch.float16 and not torch.isfinite(gnorm):
            # A non-finite gradient (e.g. after the GPU was interrupted by sleep) would turn every weight into NaN.
            # Skip this update instead; stop if it keeps happening. (float16 runs: GradScaler already skips them.)
            optimizer.zero_grad(set_to_none=True)
            bad_steps += 1
            print(f"   WARNING: non-finite gradient at step {step}, update skipped ({bad_steps} so far)")
            if bad_steps >= 20:
                raise SystemExit("STOPPED: 20 non-finite gradients. The saved checkpoint is untouched; "
                                 "restart the same command to resume from it.")
            continue
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        # ---- evaluation, logging, checkpoints ----
        if step % args.eval_interval == 0 or step == args.max_steps:
            dt = time.time() - t0
            tok_s = tokens_per_step * args.eval_interval / dt
            losses = evaluate(model, eval_batchers, args.eval_iters, ctx)
            eta_min = (args.max_steps - step) * tokens_per_step / tok_s / 60
            print(f"step {step:6d} | lr {lr:.2e} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| aux {last_aux:.3f} | {tok_s/1e3:.0f}k tok/s | ETA {eta_min:.0f} min")
            log.writerow([step, step * tokens_per_step, f"{lr:.6g}", f"{losses['train']:.5f}",
                          f"{losses['val']:.5f}", f"{last_aux:.5f}", f"{last_z:.5f}", f"{tok_s:.0f}"])
            log_f.flush()
            if not (math.isfinite(losses["val"]) and math.isfinite(losses["train"])):
                # never overwrite a good checkpoint with a broken model
                raise SystemExit(f"STOPPED at step {step}: the loss is not a number (NaN). Nothing was saved, so the "
                                 f"last checkpoint is still good: run the same command again to resume from it.")
            if losses["val"] < best_val:
                best_val = losses["val"]
                if args.save_best:
                    save(os.path.join(args.out_dir, "ckpt_best.pt"), full=False)
            # resumable checkpoint at most every resume_every_min minutes (a disconnect loses at most that much)
            if step < args.max_steps and time.time() - last_full >= 60 * args.resume_every_min:
                if save(ckpt_path, full=True):
                    last_full = time.time()
                    print(f"   saved resumable checkpoint at step {step}")
            t0 = time.time()
        if step % args.snapshot_interval == 0:
            save(os.path.join(args.out_dir, f"snapshot_step{step:06d}.pt"), full=False, half=True)
        if step % (args.eval_interval * 4) == 0 or step == args.max_steps:
            print("   sample:", sample_text(model, tok, device, eot_id).replace("\n", " ")[:400])
            t0 = time.time()

    log_f.close()
    save_until_done(state(full=False), ckpt_path)    # finished: weights only (float32), no optimizer state
    if not args.no_resume:
        write_json(done_path, {"step": step, "best_val": best_val})     # later notebooks use only finished runs
    release()
    print(f"Done. Best val loss {best_val:.4f} (perplexity {math.exp(best_val):.2f})")


if __name__ == "__main__":
    main()
