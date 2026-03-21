# GPT Token Prediction

A character-level GPT language model built from scratch using PyTorch, following Andrej Karpathy's tutorial, heavily inspired by the architecture that powers models like GPT-2 and GPT-3.

An optimized version (v2) was also implemented applying: **Fused QKV projection**, **Flash Attention** (`F.scaled_dot_product_attention`), **GELU** activation, **Gradient Clipping**, **Mixed Precision** (float16), and a **Cosine LR Scheduler**.

Baseline loss dropped from **4.41 → 1.83** val loss over 5000 steps — see the [experiment section](#experiment-baseline-vs-optimized-5000-steps-t4-gpu) for a full comparison and analysis of why the optimized version requires longer training to show its benefits.

# Building nanoGPT — The Big Picture

We have this architecture to training a GPT-2

```
Input (Tokens) 
  │
  ▼
[ Token + Positional Embedding ]
  │
  ▼
┌──────────────────────────────┐
│  LayerNorm                   │
│      ↓                       │
│  Multi-Head Attention        │  (Lặp lại N lần)
│      ↓                       │
│  Residual Connection (+x)    │
│      ↓                       │
│  LayerNorm                   │
│      ↓                       │
│  FeedForward (MLP)           │
│      ↓                       │
│  Residual Connection (+x)    │
└──────────────────────────────┘
  │
  ▼
[ Final LayerNorm ]
  │
  ▼
[ Linear Head ] → [ Softmax ] → Output (Xác suất từ kế tiếp)
```
## What is it doing?

**Goal**: Given a sequence of characters, predict the next character → generate Shakespeare-like text.

```
Input:  "To be or not to be, t"
Output: "h"  (predicts next char)
→ Repeat thousands of times → generate full paragraphs
```

---

## The Journey: Bigram → nanoGPT

### Step 1: Bigram (Dumb Baseline)

```
Current token → Lookup table → Next token
```

- Only looks at **1 token** to predict the next one — no context
- Result: garbage text, but proves the pipeline works

### Step 2: Add Self-Attention (The Core Upgrade)

```
All previous tokens → Q,K,V → Weighted mix → Better prediction
```
- Each token **asks** (Q) and **answers** (K) → compute relevance scores
- Tokens gather information from **all** previous tokens, not just the last one
- Masked so tokens **can't cheat** by looking at the future

### Step 3: Stack into Transformer Blocks (Depth)

```
Input x
  │
  ▼
LayerNorm → Attention → + x (residual)    ← Bước 1
  │
  ▼
LayerNorm → FeedForward → + x (residual)  ← Bước 2
  │
  ▼
Output x
```

### Step 4: Full nanoGPT

```
Characters → Token Embed + Position Embed → 4 Blocks → LayerNorm → Linear → Prediction
```

---

## Architecture

```
"To be or not"
      │
      ▼
┌─────────────────────────┐
│  Token Embedding        │  What is each character?
│  + Position Embedding   │  Where is it in the sequence?
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  Transformer Block ×4   │
│  ┌───────────────────┐  │
│  │ Multi-Head         │  │  4 heads look at different patterns
│  │ Self-Attention     │  │  Q×Kᵀ/√d → softmax → ×V
│  └───────────────────┘  │
│  ┌───────────────────┐  │
│  │ FeedForward        │  │  64 → 256 → 64 (expand, think, compress)
│  └───────────────────┘  │
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  Linear Head            │  64 dims → 65 chars (vocab)
-> logits 
-> softmax -> probs
-> multinomal sampling
└─────────┬───────────────┘
          ▼
    "t" → predict "o"
```

---

## Training in One Sentence

> Repeatedly: grab random text chunks → predict next char at every position → measure error (cross-entropy) → adjust all weights via backprop → repeat 5000 times.

---


## Experiment: Baseline vs Optimized (5000 steps, T4 GPU)

### What changed in the Optimized version?

| Change | Baseline (v1) | Optimized (v2) |
|---|---|---|
| Attention | Manual Q/K/V | Fused QKV + Flash Attention |
| Activation | ReLU | GELU |
| Learning Rate | Fixed `1e-3` | Cosine decay `1e-3 → 1e-4` |
| Dropout | `0.0` | `0.1` |
| Gradient Clipping | ❌ | `max_norm=1.0` |
| Mixed Precision | ❌ |  float16 autocast |

---

### Loss Comparison at 5000 steps

| Step | v1 Train | v1 Val | v2 Train | v2 Val |
|------|----------|--------|----------|--------|
| 0 | 4.41 | 4.40 | 4.40 | 4.40 |
| 1000 | 2.10 | 2.13 | 2.10 | 2.12 |
| 2500 | 1.81 | 1.94 | 1.85 | 1.95 |
| 4999 | **1.66** | **1.83** | **1.74** | **1.87** |

**Winner: v1 (Baseline)** — lower val loss by ~0.04 at the same number of steps.

---

### Why did v2 (Optimized) lose despite having more techniques?

#### Reason 1: Cosine LR Scheduler decayed too fast for only 5000 steps

The scheduler was designed for long training runs (50k+ steps). With only 5000 steps, LR had already dropped to `5.5e-4` by step 2500 — half of its starting value — causing the model to slow down learning too early.

```
step    0: lr = 1.00e-03  ← full speed
step 2500: lr = 5.50e-04  ← already half, model slowing down
step 4000: lr = 1.86e-04  ← nearly frozen
step 4999: lr = 1.00e-04  ← effectively stopped learning
```

> Fix: Use `T_max = max_iters` and more training steps (10k+), or simply use a fixed LR for short runs.

#### Reason 2: `dropout=0.1` hurts with only 5000 steps

Dropout randomly silences 10% of neurons during training, adding noise that requires more steps to overcome. With a small dataset (tiny Shakespeare, ~1M chars) and only 5000 steps, dropout slows convergence without having enough time to provide regularization benefits.

> Fix: Keep `dropout=0.0` for short training runs, or use `0.05` at most.

#### What v2 actually improved (not visible in 5000-step loss)

- **Speed**: Flash Attention reduces memory from O(T²) to O(T), faster on GPU
- **Stability**: Gradient clipping prevents loss spikes when scaling up
- **Scalability**: These techniques shine at larger model sizes and longer training

---

### Generated Text Comparison

**v1 output (val loss 1.83):**
```
KING RICHARD II:
Shal lifest made to bub, to take Our my dagatants:
Whith foul his vetward that a endrer, my fears' to zorm heavens...

WARWICK:
Welll now, and thus quechiry: there's speak you love.
```

**v2 output (val loss 1.87):**
```
LENA:
Where creess in himsess folet I Pame your lience;
This I prase and your spetcileggerving stignt is near...

JULI:
And hus harselford strain, ye.
```

Both outputs show correct Shakespeare formatting (character names, line breaks, punctuation) with misspelled words expected from a char-level model.

---

### Conclusion

> The optimizations in v2 (Fused QKV, Flash Attention, GELU, Mixed Precision) are **infrastructure improvements** — they make training faster and more memory efficient without changing the final loss at equal steps. However, the **hyperparameter changes** (Cosine LR + dropout=0.1) introduced in the same version hurt convergence within the 5000-step budget. To see the real benefit of v2, either train for 10k+ steps or revert just the LR and dropout settings.
