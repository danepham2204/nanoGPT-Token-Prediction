# GPT Token Prediction

A character-level GPT language model built from scratch using PyTorch, following Andrej's tutorial, heavily inspired by the architecture that powers models like GPT-2 and GPT-3. 

Loss dropped from **4.41 → 1.66** ( 5000 times trained with other parameter in code) — model learned average of 62%.

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


## Result

### Loss Progression (~7 min training on Colab, 0.21M parameters)

| Phase | Steps | Train Loss | Val Loss |
|-------|-------|-----------|---------|
| Start | 0 | 4.41 | 4.40 |
| Early | 500 | 2.30 | 2.31 |
| Mid | 2000 | 1.89 | 1.99 |
| Late | 4000 | 1.72 | 1.86 |
| **Final** | **4999** | **1.66** | **1.82** |



### Generated Output Sample

```
FlY BOLINGHARD:
Nay, humbract; it contes too
must encleming and the second; and say life;
In enter all I are and those it;
Give out of your I'll tom them nither,
One these is news it cy rege;
What Naying well and Burryres an fear?

OXITVOHN MONFIUS:
O is my mily.

LEONTES:
Geve worman:
But guontt not; do spost I vour have well...
```

### Observations

- ✅ Correct Shakespeare format: character names, line breaks, punctuation
- ✅ Grammatical sentence structure with subjects and verbs
- ✅ Mixed multiple characters (ROMEO, DUKE OF YORK, POMPEY...)
- ❌ Many misspelled words ("humbract", "encleming") — expected for character-level model
- ⚠️ Slight overfitting: train loss `1.66` < val loss `1.82` (gap ~0.16)

