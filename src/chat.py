"""
Team NOVA - AFML Project
chat.py: talk to a fine-tuned NovaLM, turn by turn (the demo for the course video).

The conversation is kept as one token sequence in the same format the model was fine-tuned on:
    <|instruction|>user text<|response|>assistant text<|endoftext|><|instruction|>next user text<|response|>...
Each reply is generated until the model writes <|endoftext|> (end of its turn). When the history no longer
fits in the context window (512 tokens for the Base model), the OLDEST turns are dropped, whole turns only.

Sampling controls (all can be changed per call):
  temperature        < 1 makes the choice more focused, > 1 more random
  top_k              only the k most likely tokens can be chosen
  repetition_penalty divides the score of tokens already used in the reply (> 1 = discourage repeating)

Usage in a notebook:
    from chat import Chat, load_chat_model
    model, tok = load_chat_model(".../runs/moe_sft/ckpt_best.pt", ".../data")
    bot = Chat(model, tok)
    print(bot.reply("Hi! What is photosynthesis?"))
    print(bot.reply("Explain it to a 10-year-old."))   # remembers the previous turn
    bot.reset()                                         # new conversation
From a terminal: python chat.py --ckpt .../ckpt_best.pt --data_dir .../data
"""

import argparse
import os

import torch

from routing_analysis import load_model


def load_chat_model(ckpt, data_dir, device=None):
    from tokenizers import Tokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return load_model(ckpt, device), Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))


class Chat:
    def __init__(self, model, tok, system=None, temperature=0.7, top_k=40, repetition_penalty=1.15,
                 max_new_tokens=200):
        self.model, self.tok = model, tok
        self.instr = tok.token_to_id("<|instruction|>")
        self.resp = tok.token_to_id("<|response|>")
        self.eot = tok.token_to_id("<|endoftext|>")
        self.system = system
        self.temperature, self.top_k = temperature, top_k
        self.repetition_penalty, self.max_new_tokens = repetition_penalty, max_new_tokens
        self.reset()

    def reset(self):
        """Start a new conversation."""
        self.turns = []                      # list of token-id lists, one per finished user+assistant turn

    def _context(self, user_ids):
        """Most recent whole turns + the new user turn, leaving room for the reply."""
        budget = self.model.cfg.block_size - self.max_new_tokens
        ctx = list(user_ids)
        if len(ctx) > budget:                # a very long message: keep its end
            ctx = [self.instr] + ctx[-(budget - 1):]
        for turn in reversed(self.turns):
            if len(turn) + len(ctx) > budget:
                break
            ctx = turn + ctx
        return ctx

    @torch.no_grad()
    def _generate(self, ctx):
        model = self.model
        model.eval()
        dev = next(model.parameters()).device
        idx = torch.tensor([ctx], device=dev)
        out = []
        for _ in range(self.max_new_tokens):
            logits = model(idx[:, -model.cfg.block_size:])[0][0, -1].float()
            if self.repetition_penalty != 1.0 and out:
                used = torch.tensor(sorted(set(out)), device=dev)
                vals = logits[used]
                logits[used] = torch.where(vals > 0, vals / self.repetition_penalty, vals * self.repetition_penalty)
            logits = logits / max(self.temperature, 1e-5)
            if self.top_k:
                kth = torch.topk(logits, min(self.top_k, logits.numel())).values[-1]
                logits[logits < kth] = -float("inf")
            nxt = int(torch.multinomial(logits.softmax(-1), 1))
            if nxt == self.eot:
                break
            out.append(nxt)
            idx = torch.cat([idx, torch.tensor([[nxt]], device=dev)], dim=1)
        return out

    def reply(self, text):
        """Send one user message, get the model's answer (the conversation history is kept)."""
        content = text.strip()
        if self.system and not self.turns:
            content = self.system.strip() + "\n\n" + content
        user_ids = [self.instr] + self.tok.encode(content).ids + [self.resp]
        answer = self._generate(self._context(user_ids))
        self.turns.append(user_ids + answer + [self.eot])
        return self.tok.decode(answer).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--repetition_penalty", type=float, default=1.15)
    a = p.parse_args()
    model, tok = load_chat_model(a.ckpt, a.data_dir)
    bot = Chat(model, tok, temperature=a.temperature, top_k=a.top_k, repetition_penalty=a.repetition_penalty)
    print("Chat with NovaLM. Type 'reset' for a new conversation, 'quit' to stop.")
    while True:
        text = input("You: ").strip()
        if text.lower() in ("quit", "exit"):
            break
        if text.lower() == "reset":
            bot.reset()
            print("(new conversation)")
            continue
        if text:
            print("NovaLM:", bot.reply(text))


if __name__ == "__main__":
    main()
