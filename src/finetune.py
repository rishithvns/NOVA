r"""
Team NOVA - AFML Project
finetune.py: instruction / chat fine-tuning (Phase 3).

Two datasets (--sft_data):
  alpaca    52k single-turn instruction/response pairs (small TinyStories model)
  smoltalk  smol-smoltalk (HuggingFaceTB): ~460k synthetic MULTI-TURN conversations made for models under
            1B parameters (used for SmolLM2-135M/360M-Instruct). Used for the Base model, so that it can hold
            a conversation (course feedback: "make an SLM you can actually converse with").

A conversation uses the same reserved tokens: <|instruction|> starts a USER turn, <|response|> starts an
ASSISTANT turn, <|endoftext|> ends each assistant turn:
    <|instruction|>Hi! Who are you?<|response|>I'm a small assistant.<|endoftext|><|instruction|>Tell me a joke.<|response|>...<|endoftext|>
    \______ user: mask 0 ______________/\___ assistant: mask 1 ___/
A system message, if present, is put in front of the first user message.

Takes a PRETRAINED checkpoint (dense or MoE) and continues training on chat data (smol-smoltalk for the
Base model; Alpaca only for the earlier small model), formatted with our reserved special tokens:

    <|instruction|>Give three tips for staying healthy.<|response|>1. Eat ... <|endoftext|>
    \_______________ prompt: loss mask = 0 ______________/ \_ response: loss mask = 1 _/

Only response tokens (and the final <|endoftext|>) count in the loss: the model learns to ANSWER
instructions, not to generate instructions. Padding tokens never count.

Also records, for the research:
  * weight snapshots every --snapshot_interval steps (routing drift DURING fine-tuning)
  * pretraining-data validation loss before/after (how much pretraining knowledge is forgotten)
  * sample responses to fixed instructions before/after fine-tuning (samples.txt)

Saving (same Drive-friendly rules as train.py): ckpt_best.pt = weights with the lowest instruction validation loss,
kept in memory and written to Drive together with the resumable ckpt_last.pt (every --resume_every_min minutes)
and at the end; DONE.json marks a finished run.

Usage:
  python finetune.py --data_dir DATA --init_ckpt RUNS/moe/ckpt_last.pt --out_dir RUNS/moe_sft
"""

import argparse
import csv
import json
import math
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
from tokenizers import Tokenizer

from model import ModelConfig, NovaLM
from train import Batcher, compact_optimizer_state, evaluate, safe_save, save_until_done, val_dev_end, write_json

EVAL_PROMPTS = [
    "Write a short story about a cat who loves to play.",
    "Tell a story about a boy and his dog.",
    "What color is the sky?",
    "Give three tips for staying happy.",
    "Write a sentence using the word 'friend'.",
]
EVAL_PROMPTS_CHAT = [
    "Hi! What can you help me with?",
    "What is photosynthesis? Explain it simply.",
    "Give me three tips for studying for an exam.",
    "Write a short poem about the sea.",
    "What is the capital of France, and what is it famous for?",
]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--init_ckpt", required=True, help="pretrained checkpoint to start from")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--aux_coef", type=float, default=0.01)
    p.add_argument("--z_coef", type=float, default=1e-3)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--snapshot_interval", type=int, default=100)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max_examples", type=int, default=None, help="for quick tests")
    p.add_argument("--sft_data", default="alpaca", choices=["alpaca", "smoltalk"])
    p.add_argument("--max_train", type=int, default=None,
                   help="smoltalk: number of training conversations to use (the full set is ~460k)")
    p.add_argument("--n_eval", type=int, default=2000, help="smoltalk: conversations for validation and for test")
    p.add_argument("--resume_every_min", type=float, default=45,
                   help="write the resumable checkpoint (and the best weights so far) to out_dir at most this often")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Data: Alpaca -> token ids + loss masks
# ----------------------------------------------------------------------------
def format_prompt(instruction, inp=""):
    prompt = "<|instruction|>" + instruction.strip()
    if inp and inp.strip():
        prompt += "\n" + inp.strip()
    return prompt + "<|response|>"


def encode_example(tok, instruction, inp, output, eot_id):
    """Return (ids, mask): mask[i] = 1 if token i is part of the response (incl. final eot)."""
    p_ids = tok.encode(format_prompt(instruction, inp)).ids
    r_ids = tok.encode(output.strip()).ids + [eot_id]
    return p_ids + r_ids, [0] * len(p_ids) + [1] * len(r_ids)


def load_instruction_data(data_dir, tok, eot_id, block_size, val_frac, seed, max_examples=None):
    """Tokenize Alpaca once and cache it in data_dir. Keeps examples that fit in block_size+1.
    Split: 5% validation (monitoring + best checkpoint), 5% TEST (touched only once, in notebook 06),
    90% train. Shuffled with a fixed seed so the split never changes."""
    cache = os.path.join(data_dir, f"alpaca_tok_{block_size}_split3.pt")
    if os.path.exists(cache) and max_examples is None:
        d = torch.load(cache, weights_only=False)
        print(f"Loaded cached instruction data: {len(d['train'])} train / {len(d['val'])} val / {len(d['test'])} test")
        return d
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    kept, too_long = [], 0
    for ex in ds:
        ids, mask = encode_example(tok, ex["instruction"], ex["input"], ex["output"], eot_id)
        if len(ids) > block_size + 1 or sum(mask) < 2:
            too_long += 1
            continue
        kept.append((ids, mask))
    random.Random(seed).shuffle(kept)
    n_val = max(1, int(len(kept) * val_frac))
    d = {"val": kept[:n_val], "test": kept[n_val : 2 * n_val], "train": kept[2 * n_val :]}
    stats = {"total": len(ds), "kept": len(kept), "dropped_too_long": too_long,
             "train": len(d["train"]), "val": len(d["val"]), "test": len(d["test"]),
             "mean_len": float(np.mean([len(i) for i, _ in kept])),
             "mean_response_len": float(np.mean([sum(m) for _, m in kept]))}
    d["stats"] = stats
    print("Instruction data:", json.dumps(stats))
    if max_examples is None:
        torch.save(d, cache)
    return d


def encode_conversation(tok, messages, sp, block_size):
    """Multi-turn chat -> (ids, mask). mask = 1 on assistant tokens and the <|endoftext|> that ends each
    assistant turn. Whole turns that do not fit in block_size + 1 tokens are cut off (the conversation is
    kept up to the last complete turn that fits). Returns None if not even one assistant turn fits."""
    instr, resp, eot = sp["<|instruction|>"], sp["<|response|>"], sp["<|endoftext|>"]
    ids, mask, system, user = [], [], "", None
    for m in messages:
        role, text = m["role"], (m.get("content") or "").strip()
        if role == "system":
            system = text
        elif role == "user":
            user = [instr] + tok.encode((system + "\n\n" + text) if system else text).ids
            system = ""
        elif role == "assistant" and user is not None:
            answer = tok.encode(text).ids + [eot]
            if len(ids) + len(user) + 1 + len(answer) > block_size + 1:
                break                                            # this turn does not fit: stop here
            ids += user + [resp] + answer
            mask += [0] * (len(user) + 1) + [1] * len(answer)
            user = None
    if sum(mask) < 2:
        return None
    return ids, mask


def load_smoltalk_data(data_dir, tok, sp, block_size, seed, max_train=None, n_eval=2000, max_examples=None):
    """smol-smoltalk: training conversations from its train split, validation and test from its test split
    (n_eval each). Tokenized once and cached in data_dir."""
    tag = f"{max_train or 'all'}"
    cache = os.path.join(data_dir, f"smoltalk_tok_{block_size}_{tag}.pt")
    if os.path.exists(cache) and max_examples is None:
        d = torch.load(cache, weights_only=False)
        print(f"Loaded cached chat data: {len(d['train'])} train / {len(d['val'])} val / {len(d['test'])} test")
        return d
    from datasets import load_dataset
    tr = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    te = load_dataset("HuggingFaceTB/smol-smoltalk", split="test")
    tr = tr.shuffle(seed=seed)
    n_tr = min(len(tr), max_examples or max_train or len(tr))
    n_ev = min(n_eval, (max_examples or n_eval), len(te) // 2)
    stats = {"train_available": len(tr), "test_available": len(te)}

    def encode_all(rows, limit):
        out, dropped, turns = [], 0, []
        for ex in rows:
            r = encode_conversation(tok, ex["messages"], sp, block_size)
            if r is None:
                dropped += 1
                continue
            out.append(r)
            turns.append(sum(1 for t in r[0] if t == sp["<|response|>"]))
            if len(out) >= limit:
                break
        return out, dropped, turns

    train, dr_tr, turns = encode_all(tr, n_tr)
    ev, dr_ev, _ = encode_all(te, 2 * n_ev)
    d = {"train": train, "val": ev[:n_ev], "test": ev[n_ev:2 * n_ev]}
    stats.update({"train": len(train), "val": len(d["val"]), "test": len(d["test"]),
                  "dropped_too_long": dr_tr + dr_ev,
                  "mean_assistant_turns": round(float(np.mean(turns)), 2),
                  "mean_len": round(float(np.mean([len(i) for i, _ in train])), 1),
                  "mean_assistant_tokens": round(float(np.mean([sum(m) for _, m in train])), 1)})
    d["stats"] = stats
    print("Chat data:", json.dumps(stats))
    if max_examples is None:
        torch.save(d, cache)
    return d


def make_batch(examples, pad_id, device):
    """Pad to the longest example. x = ids[:-1], y = ids[1:], loss_mask = response & not pad."""
    L = max(len(ids) for ids, _ in examples)
    B = len(examples)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    msk = torch.zeros((B, L), dtype=torch.long)
    for b, (i, m) in enumerate(examples):
        ids[b, : len(i)] = torch.tensor(i)
        msk[b, : len(m)] = torch.tensor(m)
    x, y, lm = ids[:, :-1].contiguous(), ids[:, 1:].contiguous(), msk[:, 1:].contiguous()
    return x.to(device), y.to(device), lm.to(device)


@torch.no_grad()
def instruction_val_loss(model, val, pad_id, device, ctx, batch_size=64):
    model.eval()
    tot, n = 0.0, 0.0
    for s in range(0, len(val), batch_size):
        x, y, lm = make_batch(val[s : s + batch_size], pad_id, device)
        with ctx:
            _, loss, _ = model(x, y, loss_mask=lm)
        w = lm.sum().item()
        tot += loss.item() * w
        n += w
    model.train()
    return tot / max(n, 1)


@torch.no_grad()
def respond(model, tok, prompts, device, eot_id, max_new=120):
    model.eval()
    out = []
    instr, resp = tok.token_to_id("<|instruction|>"), tok.token_to_id("<|response|>")
    for pr in prompts:
        idx = torch.tensor([[instr] + tok.encode(pr.strip()).ids + [resp]], device=device)
        gen = model.generate(idx, max_new, temperature=0.7, top_k=40, eot_id=eot_id)
        text = tok.decode(gen[0, idx.shape[1]:].tolist())
        out.append(f"### {pr}\n{text.strip()}\n")
    model.train()
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = args.device

    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    st = meta["special_tokens"]
    eot_id, pad_id = st["<|endoftext|>"], st["<|pad|>"]
    tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))

    ck = torch.load(args.init_ckpt, map_location=device, weights_only=False)
    c = dict(ck["config"])
    c["moe_layers"] = tuple(c["moe_layers"])
    cfg = ModelConfig(**c)
    model = NovaLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    is_moe = len(cfg.moe_layers) > 0

    if args.sft_data == "smoltalk":
        data = load_smoltalk_data(args.data_dir, tok, st, cfg.block_size, args.seed, args.max_train,
                                  args.n_eval, args.max_examples)
        prompts = EVAL_PROMPTS_CHAT
    else:
        data = load_instruction_data(args.data_dir, tok, eot_id, cfg.block_size, args.val_frac,
                                     args.seed, args.max_examples)
        prompts = EVAL_PROMPTS
    train, val = data["train"], data["val"]
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    max_steps = int(args.epochs * steps_per_epoch)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        dtype, ctx = torch.float32, nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        prog = (step - args.warmup_steps) / max(1, max_steps - args.warmup_steps)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(prog, 1.0)))

    val_path = os.path.join(args.data_dir, "val.bin")      # development half only: the test half stays untouched
    ts_val = Batcher(val_path, cfg.block_size, 32, device, 999, limit=val_dev_end(val_path, eot_id))

    ckpt_path = os.path.join(args.out_dir, "ckpt_last.pt")
    step, best = 0, float("inf")
    if os.path.exists(ckpt_path) and not args.no_resume:
        r = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "optimizer" not in r:                          # finished runs keep weights only
            print(f"Fine-tuning already finished at step {r['step']} (best {r['best_val']:.4f}); nothing to do.")
            if not os.path.exists(os.path.join(args.out_dir, "DONE.json")):
                write_json(os.path.join(args.out_dir, "DONE.json"), {"step": r["step"], "best_val": r["best_val"]})
            return
        model.load_state_dict(r["model"])
        opt.load_state_dict(r["optimizer"])
        scaler.load_state_dict(r["scaler"])
        step, best = r["step"], r["best_val"]
        print(f"Resumed fine-tuning from step {step}")

    def state(full=False, half=False, weights=None):
        sd = weights if weights is not None else model.state_dict()
        if half:                                          # snapshots: float16 weights, half the size
            sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
        s = {"model": sd, "config": c, "step": step, "best_val": best,
             "init_ckpt": args.init_ckpt, "sft_data": args.sft_data}
        if full:
            s.update(optimizer=compact_optimizer_state(opt.state_dict()), scaler=scaler.state_dict())
        return s

    def save(path, full=True, half=False):            # atomic; warns instead of crashing when Drive is full
        return safe_save(state(full, half), path)

    # The best weights so far live in CPU memory and reach Drive together with the resumable checkpoint (and at the
    # end), so the two files on Drive always agree and the best checkpoint is not rewritten at every improvement.
    best_path = os.path.join(args.out_dir, "ckpt_best.pt")
    best_weights, best_step, best_dirty = None, step, False

    def flush_best():
        nonlocal best_dirty
        if best_dirty and save_best_file():
            best_dirty = False
        return not best_dirty

    def save_best_file():
        s = state(weights=best_weights)
        s["step"] = best_step
        return safe_save(s, best_path)

    log_path = os.path.join(args.out_dir, "log.csv")
    new_log = step == 0
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_f)

    if step == 0:
        log.writerow(["step", "epoch", "lr", "train_loss", "instr_val_loss", "pretrain_val_loss", "aux_loss"])
        v0 = instruction_val_loss(model, val, pad_id, device, ctx)
        ts0 = evaluate(model, {"val": ts_val}, 20, ctx)["val"]
        log.writerow([0, 0, 0, "", f"{v0:.5f}", f"{ts0:.5f}", ""])
        save(os.path.join(args.out_dir, "snapshot_step000000.pt"), full=False, half=True)   # = pretrained weights
        with open(os.path.join(args.out_dir, "samples.txt"), "w") as f:
            f.write("=== BEFORE instruction tuning ===\n\n" + respond(model, tok, prompts, device, eot_id))
        print(f"Before fine-tuning: instruction val loss {v0:.4f} | pretraining-data val loss {ts0:.4f}")
        json.dump({"args": vars(args), "data": data.get("stats", {})},
                  open(os.path.join(args.out_dir, "config.json"), "w"), indent=2)

    print(f"{'MoE' if is_moe else 'Dense'} | {len(train)} train examples | {max_steps} steps "
          f"({args.epochs} epochs x {steps_per_epoch})")

    order = list(range(len(train)))
    model.train()
    last_full = time.time()
    t0, last_loss, last_aux = time.time(), float("nan"), float("nan")
    while step < max_steps:
        epoch_pos = (step * args.batch_size) % len(train)
        if epoch_pos < args.batch_size:                     # new epoch -> reshuffle (seeded by epoch)
            random.Random(args.seed + step // steps_per_epoch).shuffle(order)
        batch = [train[order[(epoch_pos + i) % len(train)]] for i in range(args.batch_size)]
        x, y, lm = make_batch(batch, pad_id, device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with ctx:
            _, loss, info = model(x, y, loss_mask=lm, valid_mask=(x != pad_id))
            total = loss
            if info["aux_loss"] is not None:
                total = total + args.aux_coef * info["aux_loss"] + args.z_coef * info["z_loss"]
                last_aux = info["aux_loss"].item()
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        last_loss = loss.item()

        if step % args.snapshot_interval == 0:
            save(os.path.join(args.out_dir, f"snapshot_step{step:06d}.pt"), full=False, half=True)
        if step % args.eval_interval == 0 or step == max_steps:
            v = instruction_val_loss(model, val, pad_id, device, ctx)
            ts = evaluate(model, {"val": ts_val}, 20, ctx)["val"]
            print(f"step {step:5d}/{max_steps} | lr {lr_at(step):.2e} | train {last_loss:.4f} | "
                  f"instr val {v:.4f} | pretrain val {ts:.4f} | {time.time() - t0:.0f}s")
            log.writerow([step, f"{step / steps_per_epoch:.3f}", f"{lr_at(step):.6g}", f"{last_loss:.5f}",
                          f"{v:.5f}", f"{ts:.5f}", f"{last_aux:.5f}"])
            log_f.flush()
            if v < best:
                best, best_step, best_dirty = v, step, True
                best_weights = {k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()}
            if step < max_steps and time.time() - last_full >= 60 * args.resume_every_min:
                if flush_best() and save(ckpt_path, full=True):   # best first: the resume point never runs ahead of it
                    last_full = time.time()
                    print(f"   saved resumable checkpoint at step {step}")

    log_f.close()
    if best_dirty:
        save_until_done({**state(weights=best_weights), "step": best_step}, best_path)
    save_until_done(state(full=False), ckpt_path)          # finished: weights only
    write_json(os.path.join(args.out_dir, "DONE.json"), {"step": step, "best_val": best})
    final = respond(model, tok, prompts, device, eot_id)
    with open(os.path.join(args.out_dir, "samples.txt"), "a") as f:
        f.write("\n=== AFTER instruction tuning ===\n\n" + final)
    print(final)
    print(f"Done. Best instruction val loss {best:.4f}")


if __name__ == "__main__":
    main()
