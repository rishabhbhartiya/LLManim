<div align="center">

<img src="llmanim.svg" alt="LLManim" width="680" />


**Reusable Manim components for creating LLM and Transformer explanation videos.**

[![PyPI version](https://img.shields.io/pypi/v/llmanim.svg)](https://pypi.org/project/llmanim/)
[![Python](https://img.shields.io/pypi/pyversions/llmanim.svg)](https://pypi.org/project/llmanim/)
[![CI](https://github.com/rishabhbhartiya/LLManim/actions/workflows/ci.yml/badge.svg)](https://github.com/rishabhbhartiya/LLManim/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python library of pre-built visual components, animations, and utilities for creating educational videos about Large Language Models and Transformer architectures using [Manim](https://www.manim.community/).

[Installation](#installation) · [Quick Start](#quick-start) · [Components](#components) · [Examples](#example-videos) · [Contributing](CONTRIBUTING.md)

</div>

---

## Installation

```bash
pip install llmanim
```

### Prerequisites

Manim requires a few system dependencies before you install:

```bash
# macOS
brew install cairo pango ffmpeg
brew install --cask mactex          # LaTeX for MathTex

# Ubuntu / Debian
sudo apt install libcairo2-dev libpango1.0-dev ffmpeg texlive-full

# Windows — install MiKTeX from https://miktex.org, then:
# pip install manim
```

Full Manim installation guide: [docs.manim.community](https://docs.manim.community/en/stable/installation.html)

### Install from source

```bash
git clone https://github.com/rishabhbhartiya/LLManim.git
cd LLManim
pip install -e ".[dev]"
```

---

## Quick Start

```python
from manim import *
from llmanim.base.shapes import MatrixBox, VectorBar, TokenBox
from llmanim.base.animations import highlight_sequence, data_flow_arrow
from llmanim.base.utils import apply_dark_theme, make_attention_weights
from llmanim.tokenization.token_box import TokenRow
from llmanim.styles.colors import TOKEN_COLOR, HIGHLIGHT_COLOR

class MyTransformerScene(Scene):
    def construct(self):
        apply_dark_theme(self)

        # 1. Show a token sequence
        row = TokenRow(["the", "cat", "sat"], token_ids=[1, 2, 3])
        self.play(row.appear())

        # 2. Show an embedding matrix
        mat = MatrixBox(rows=4, cols=6, color=TOKEN_COLOR).shift(DOWN * 2)
        self.play(mat.fill_anim())

        # 3. Connect them with an animated arrow
        arrow, anim = data_flow_arrow(row, mat, label="embed")
        self.add(arrow)
        self.play(anim)

        # 4. Highlight components
        self.play(highlight_sequence(row.boxes))
        self.play(mat.highlight_row(2))
        self.wait(1)
```

Render it:

```bash
manim -pql my_scene.py MyTransformerScene   # fast preview
manim -pqh my_scene.py MyTransformerScene   # 1080p final
```

---

## Example Videos

Five complete video scripts are included in `examples/`:

| # | File | Scene | Topic |
|---|------|-------|-------|
| 1 | `01_what_is_a_token.py` | `WhatIsAToken` | Tokenization basics |
| 2 | `02_token_embeddings.py` | `TokenEmbeddings` | Embedding lookup |
| 3 | `03_attention_mechanism.py` | `AttentionMechanism` | Q, K, V & softmax |
| 4 | `04_multi_head_attention.py` | `MultiHeadAttention` | Multiple attention heads |
| 5 | `05_full_forward_pass.py` | `FullForwardPass` | Complete forward pass |

### Video 1 — What is a Token?

<img src="videos/WhatIsAToken.gif" alt="What is a Token?" width="720" />

### Video 2 — Token Embeddings

<img src="videos/TokenEmbeddings.gif" alt="Token Embeddings" width="720" />

### Video 3 — Attention Mechanism

<img src="videos/AttentionMechanism.gif" alt="Attention Mechanism" width="720" />

### Video 4 — Multi-Head Attention

<img src="videos/MultiHeadAttention.gif" alt="Multi-Head Attention" width="720" />

### Video 5 — Full Forward Pass

<img src="videos/FullForwardPass.gif" alt="Full Transformer Forward Pass" width="720" />

```bash
# Clone to access examples (they're not installed with pip)
git clone https://github.com/rishabhbhartiya/LLManim.git
cd LLManim

manim -pql examples/01_what_is_a_token.py WhatIsAToken     # preview
manim -pqh examples/05_full_forward_pass.py FullForwardPass # 1080p
manim -pqk examples/05_full_forward_pass.py FullForwardPass # 4K
```

---

## Components

### Base Shapes — `llmanim.base.shapes`

| Class | Description | Key Methods |
|-------|-------------|-------------|
| `MatrixBox(rows, cols)` | Colored grid for weight matrices | `.highlight_row(r)`, `.highlight_col(c)`, `.fill_anim()` |
| `VectorBar(dim, values)` | Colored strip for embedding vectors | `.pulse()`, `.highlight_dim(i)`, `.add_value_labels()` |
| `TokenBox(text, token_id)` | Rounded rectangle for a single token | `.highlight()`, `.shake()`, `.pop_in()`, `.pop_out()` |
| `LabeledBlock(label)` | Named rectangular block | `.highlight()`, `.expand_to(mob)` |
| `SoftmaxCurve(logits)` | Probability bar chart | `.set_temperature(T)`, `.show_values()` |
| `MathLabel(latex)` | LaTeX equation label | `.write()`, `.unwrite()` |
| `ConnectionLine(srcs, tgts)` | Fully connected layer lines | `.draw()`, `.activate(i,j)`, `.pulse_all()` |

### Animations — `llmanim.base.animations`

| Function | Description |
|----------|-------------|
| `flow_through(mob, path)` | Animate an object along a path |
| `matrix_multiply_anim(A, B, C, scene)` | Step-by-step matrix multiplication |
| `highlight_sequence(boxes)` | Flash boxes one by one |
| `attention_flow(tokens, weights)` | Weighted attention arrows between tokens |
| `data_flow_arrow(A, B, label)` | Animated labeled arrow between mobjects |
| `equation_reveal(terms, scene)` | Build an equation term by term |
| `pulse_glow(mob)` | Expanding glow effect |
| `zoom_into(mob, scene)` | Zoom in and back out |
| `forward_pass_pulse(blocks)` | Light pulse travelling through layers |

### Utilities — `llmanim.base.utils`

| Function | Description |
|----------|-------------|
| `softmax(x, temperature)` | Numerically stable softmax |
| `scaled_dot_product(Q, K, V, mask)` | Full attention computation |
| `causal_mask(seq_len)` | Upper-triangular causal mask |
| `sinusoidal_pe(seq_len, d_model)` | Sinusoidal positional encoding |
| `make_attention_weights(seq_len, n_heads)` | Generate demo attention weights |
| `make_token_sequence(words)` | Create a demo token sequence |
| `value_to_color(v)` | Map a float to a Manim color |
| `attention_palette(weights)` | Color palette from attention weights |
| `apply_dark_theme(scene)` | Set dark background on a Scene |

### Tokenization — `llmanim.tokenization.token_box`

| Class | Description |
|-------|-------------|
| `TokenRow(tokens, token_ids)` | A horizontal row of token boxes |
| `TokenBox(text, token_id)` | Individual token visualization |
| `SpecialTokenBox(text)` | Special tokens like `[CLS]`, `[SEP]` |
| `BPESequence` | Byte-pair encoding visualization |
| `VocabTable` | Vocabulary lookup table |
| `ContextWindowBar` | Context window capacity indicator |

---

## Color System

All colors are in `llmanim.styles.colors`:

| Constant | Hex | Used for |
|----------|-----|----------|
| `TOKEN_COLOR` | `#4CAF50` | Tokens, raw input |
| `EMBEDDING_COLOR` | `#2196F3` | Embeddings, weight matrices |
| `ATTENTION_COLOR` | `#9C27B0` | Attention mechanisms |
| `FFN_COLOR` | `#FF9800` | Feed-forward networks |
| `NORM_COLOR` | `#00BCD4` | Layer normalization |
| `OUTPUT_COLOR` | `#F44336` | Output logits |
| `QUERY_COLOR` | `#E91E63` | Query (Q) vectors |
| `KEY_COLOR` | `#3F51B5` | Key (K) vectors |
| `VALUE_COLOR` | `#009688` | Value (V) vectors |
| `HIGHLIGHT_COLOR` | `#FFEB3B` | Highlights, focus |
| `DIM_COLOR` | `#555555` | Dimmed / background elements |

---

## Project Structure

```
LLManim/
├── llmanim/           # Installed package (import as llmanim)
│   ├── attention/               # QKV projection, scores, output, multi-head
│   ├── base/                    # shapes.py, animations.py, utils.py
│   ├── blocks/                  # transformer_block.py
│   ├── embeddings/              # embedding_lookup.py, positional_encoding.py
│   ├── feedforward/             # ffn.py
│   ├── normalization/           # layer_norm.py
│   ├── output/                  # logits_sampling.py
│   ├── styles/                  # colors.py, constants.py, fonts.py
│   ├── tokenization/            # token_box.py
│   └── full_model.py
├── examples/                    # Standalone video scripts (not installed)
├── videos/                      # Output GIFs for all example scenes
└── tests/
```

---

## Video Production Workflow

1. **Plan** — write your script outline and narration
2. **Import** — pick components from `llmanim`
3. **Sketch** — plan scene layout on paper
4. **Code** — build scenes incrementally, preview with `-pql`
5. **Render** — final export with `-pqh` or `-pqk`
6. **Voiceover** — record audio to match animation timing
7. **Edit** — combine in your video editor of choice

---

## Contributing

Contributions are welcome — new components, bug fixes, example videos, or docs improvements.
See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

## License

[MIT](LICENSE) © [Rishabh Bhartiya](https://github.com/rishabhbhartiya)

---

<div align="center">
Built with <a href="https://www.manim.community/">Manim Community Edition</a>
</div>