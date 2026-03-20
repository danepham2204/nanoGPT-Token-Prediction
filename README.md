# GPT Token Prediction

A character-level GPT language model built from scratch using PyTorch, following Andrej's tutorial, heavily inspired by the architecture that powers models like GPT-2 and GPT-3.

# Building nanoGPT — The Big Picture

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

## Key Building Blocks Summary

| Block | What it does | Why needed |
|-------|-------------|-----------|
| **Embedding** | Char → vector | Computers need numbers, not letters |
| **Position Embed** | Encode where in sequence | Attention has no built-in sense of order |
| **Self-Attention** | Tokens communicate | "Who should I pay attention to?" |
| **FeedForward** | Tokens think alone | Process the gathered information |
| **Residual (`x + f(x)`)** | Skip connections | Prevents signal loss in deep networks |
| **LayerNorm** | Normalize values | Keeps training stable |
| **Softmax** | Scores → probabilities | Need proper distribution to sample from |
| **Causal Mask** | Block future tokens | Can't cheat during training |
