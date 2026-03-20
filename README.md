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

-  Correct Shakespeare format: character names, line breaks, punctuation
-  Grammatical sentence structure with subjects and verbs
-  Mixed multiple characters (ROMEO, DUKE OF YORK, POMPEY...)
- ❌ Many misspelled words ("humbract", "encleming") — expected for character-level model
- ⚠️ Slight overfitting: train loss `1.66` < val loss `1.82` (gap ~0.16)


## Full text generation

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
But guontt not; do spost I vour have well;
Not and go the rivisher's become,
And alight, upon Crame be with the On man.

Roman:
What I would and Capolicioual;
And wife must he awour,
Butcousins the solle with he twomment. Gefore hild you sure
That state my not.

DUKE OF YORK:
My surnt not I have too gentle men
Comily comport's that him; I cannot this your
house. But as bathol! and now your and;
Which-suppy will to coursein to shall her spersend,
That you holk all gentled to plartes no mune in en slaicsion,
But
Thmal, but terruly friend
Ristom with the rigess and wilt tentry:
I dry that kisspy guase, we mine! crut while with up,
I som fries that neish he pray, if,
Thom the hre seinged fleby devir begom as goody.
Go as thee, thou would may night.

ROMEO:
It gantle behone, thy lasbeet, him our sitive on;
The now to be, all gokss noblambsties. joy to you would do to the woold,
Northy will your sould in him, Andrend.

LE:
My, wense what I will betters, that them end all the sposse is seeess,
I Tostry experirts livants you great?
I shalk I suort set, for this glied.
The some it, men vanty lieht. Murst; or us Volner, still;
I wear his crumpurats there suiless Edwift a thoughanted to your ground.
Where-be in his is
Hard tode toble anoced me the ords,
Wonestiful be sweet flough. were you, where 'twon enmer, 'word.

POMPEY:
Whus bot azy houth this sele yourders?

POLFORD NORK:
Yet O, sapewer, conted, so, good agion mise thy done
on his iffather Befole wefpate,
And hrow I teass in I knounged my spite
but age so sucalf me with non your
and:
As one thums of the slive righanneds:
Has then that with, and wein that we sterp'd hurse comison toOH!

SICHAM:
What'm, I have it:
Twere I pear news,
Twas wha

