# Design choices and the evidence behind them

Each architecture and training choice in NovaLM, why it was made, and what our own runs show about it.

| Choice | Why | Evidence from our runs |
|---|---|---|
| **Decoder-only transformer, 12 blocks × 768** | GPT-2-small size class: the largest model one 8 GB laptop GPU can train on 393M tokens in about 6 hours. | Dense: 88.1M parameters, 18.2k tokens/s, val loss 3.395. |
| **Own BPE tokenizer, 16,384 tokens** | Trained on our data mix; a small vocabulary keeps the embedding table to 12.6M of the 88M parameters and fits token ids in `uint16`. Includes our chat tokens. | Avg 824 tokens per document; reproducible rebuild gave identical token counts. |
| **RoPE instead of learned positions** | Encodes relative position by rotating queries and keys; no position table to learn, and generalises better to positions seen rarely. Standard in Llama/Mistral. | Training stable from step 0; no separate position parameters. |
| **RMSNorm, pre-norm** | Cheaper than LayerNorm (no mean subtraction) and equally stable; pre-norm keeps gradients well-behaved in deep stacks. | Train/val loss within 0.01 throughout. |
| **SwiGLU feed-forward** | Gated activation gives lower loss than GELU at equal parameters (used in Llama, PaLM). | — (not ablated) |
| **Grouped-query attention, 12 query / 4 key-value heads** | Each key/value head is shared by 3 query heads: about one third of the key/value parameters and KV-cache memory, with negligible quality loss (Llama-2 70B, Mistral). | Fits the 8 GB budget with batch 8 × 512 tokens. |
| **QK-norm** | Normalising queries and keys keeps the attention logits bounded, which keeps training stable; used in recent models such as OLMo-2, Qwen3 and Gemma-3. | — (not ablated) |
| **Tied input/output embeddings** | Saves 12.6M parameters (14% of the model) and helps small models. | — |
| **MoE: 8 experts, top-2** | Top-2 with half-width experts gives exactly the dense model's active compute, so the comparison is fair. With top-1, the softmax weight of the single chosen expert is 1 and the router receives no gradient from the language-model loss. Top-2 is used by GShard and Mixtral, and it defines our flip margin z(2) − z(3). | MoE 3.360 vs dense 3.395 at 88.1M active parameters; test perplexity 29.4 vs 30.5. |
| **Half-width experts (1,024 vs 2,048)** | Two half-width experts = one full feed-forward layer of compute per token. | Active parameters 88.12M vs 88.10M dense. |
| **MoE in blocks 3, 7, 11 (every 4th block)** | Spreads routing through early, middle and late depth with only 3 routers to study; fewer MoE layers keep the 8 GB memory budget. | Placement ablation (blocks 1, 2, 5): 3.3628 vs 3.3600, a 0.003 difference — placement does not change quality at this scale. |
| **Load-balancing loss, weight 0.01** | Switch Transformer's standard value; pushes expert usage towards uniform without dominating the language-model loss. | Without it: one expert takes 44% of tokens in layer 10, an expert in layer 2 is almost never used, aux loss 1.90 vs 1.01, val loss 3.368 vs 3.360. |
| **Router z-loss, weight 1e-3** | ST-MoE's standard value; penalises large router logits for numerical stability. | No router instabilities in any run. |
| **AdamW, lr 6e-4, warmup 300, cosine to 6e-5, weight decay 0.1, clip 1.0** | Standard GPT-style recipe for this model size. | Smooth loss curves for all four pretraining runs. |
| **65,536 tokens per step (8 × 16 accumulation × 512)** | Large effective batch on an 8 GB GPU through gradient accumulation. | — |
| **bfloat16** | Halves memory and needs no loss scaling on the RTX 5050. | — |
| **Chat loss on assistant tokens only** | The model should learn to answer, not to copy the user's message. | Chat loss 2.75 → 1.91 (MoE). |
