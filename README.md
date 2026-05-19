# manim_transformer
> Reusable Manim components for creating LLM and Transformer explanation videos.

A Python library that provides pre-built visual components, animations, and utilities for creating educational videos about Large Language Models and Transformer architectures using Manim.

---

## 📁 Project Structure

```
LLManim/
│
├── manim_transformer/           # Main package (gets installed via pip)
│   ├── attention/              # Attention mechanism components
│   │   ├── attention_output.py
│   │   ├── attention_score.py
│   │   ├── multi_head.py
│   │   └── qkv_projection.py
│   │
│   ├── base/                   # Core visual classes & utilities
│   │   ├── shapes.py          # MatrixBox, VectorBar, TokenBox, etc.
│   │   ├── animations.py      # Reusable animation functions
│   │   └── utils.py           # Math helpers (softmax, attention weights, etc.)
│   │
│   ├── blocks/                 # Transformer building blocks
│   │   └── transformer_block.py
│   │
│   ├── embeddings/            # Embedding & positional encoding
│   │   ├── embedding_lookup.py
│   │   └── positional_encoding.py
│   │
│   ├── feedforward/           # Feed-forward network components
│   │   └── ffn.py
│   │
│   ├── normalization/         # Layer normalization
│   │   └── layer_norm.py
│   │
│   ├── output/                # Output layer & sampling
│   │   └── logits_sampling.py
│   │
│   ├── styles/                # Colors, fonts, constants
│   │   ├── colors.py
│   │   ├── constants.py
│   │   └── fonts.py
│   │
│   ├── tokenization/          # Tokenization components
│   │   └── token_box.py
│   │
│   └── full_model.py          # Complete transformer model
│
└── examples/                   # Sample video scripts (not installed)
    ├── 01_what_is_a_token.py
    ├── 02_token_embeddings.py
    ├── 03_attention_mechanism.py
    ├── 04_multi_head_attention.py
    └── 05_full_forward_pass.py
```

---

## ⚙️ Installation

### Install from PyPI (coming soon)

```bash
pip install manim-transformer
```

### Install from source

```bash
git clone https://github.com/yourusername/LLManim.git
cd LLManim
pip install -e .
```

### Prerequisites

```bash
# Manim and dependencies
pip install manim numpy

# LaTeX (required for MathTex):
# macOS:   brew install --cask mactex
# Linux:   sudo apt install texlive-full
# Windows: install MiKTeX from miktex.org
```

---

## 🎬 Quick Start

### Run Example Videos

```bash
# Clone the repository to access examples
git clone https://github.com/yourusername/LLManim.git
cd LLManim

# Preview (fast, low quality, opens immediately)
manim -pql examples/01_what_is_a_token.py WhatIsAToken

# Medium quality
manim -pqm examples/03_attention_mechanism.py AttentionMechanism

# Final render (1080p)
manim -pqh examples/05_full_forward_pass.py FullForwardPass

# 4K render
manim -pqk examples/05_full_forward_pass.py FullForwardPass
```

---

## 🧩 Using the Library in Your Own Scripts

### Basic Example

```python
from manim import *
from manim_transformer.base.shapes import MatrixBox, VectorBar, TokenBox
from manim_transformer.base.animations import highlight_sequence, data_flow_arrow
from manim_transformer.base.utils import apply_dark_theme, make_attention_weights
from manim_transformer.tokenization.token_box import TokenRow
from manim_transformer.styles.colors import TOKEN_COLOR, HIGHLIGHT_COLOR

class MyTransformerScene(Scene):
    def construct(self):
        apply_dark_theme(self)

        # 1. Show a token sequence
        tokens = ["the", "cat", "sat"]
        ids = [1, 2, 3]
        row = TokenRow(tokens, token_ids=ids)
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

---

## 📦 Available Components

### Base Shapes (`manim_transformer.base.shapes`)

| Class | Description | Key Methods |
|---|---|---|
| `MatrixBox(rows, cols)` | Colored grid for matrices | `.highlight_row(r)`, `.highlight_col(c)`, `.fill_anim()` |
| `VectorBar(dim, values)` | Colored strip for vectors | `.pulse()`, `.highlight_dim(i)`, `.add_value_labels()` |
| `TokenBox(text, token_id)` | Rounded rectangle for tokens | `.highlight()`, `.shake()`, `.pop_in()`, `.pop_out()` |
| `LabeledBlock(label)` | Named rectangular block | `.highlight()`, `.expand_to(mob)` |
| `SoftmaxCurve(logits)` | Bar chart for probabilities | `.set_temperature(T)`, `.show_values()` |
| `MathLabel(latex)` | LaTeX equation label | `.write()`, `.unwrite()` |
| `ConnectionLine(srcs, tgts)` | Fully connected layer lines | `.draw()`, `.activate(i,j)`, `.pulse_all()` |

### Animations (`manim_transformer.base.animations`)

| Function | Description |
|---|---|
| `flow_through(mob, path)` | Animate object along a path |
| `matrix_multiply_anim(A, B, C, scene)` | Step-by-step matrix multiplication |
| `highlight_sequence(boxes)` | Flash boxes sequentially |
| `attention_flow(tokens, weights)` | Weighted attention arrows |
| `data_flow_arrow(A, B, label)` | Animated labeled arrow |
| `equation_reveal(terms, scene)` | Build equation incrementally |
| `pulse_glow(mob)` | Expanding glow effect |
| `zoom_into(mob, scene)` | Zoom in and out |
| `forward_pass_pulse(blocks)` | Light pulse through layers |

### Utilities (`manim_transformer.base.utils`)

| Function | Description |
|---|---|
| `softmax(x, temperature)` | Numerically stable softmax |
| `scaled_dot_product(Q, K, V, mask)` | Full attention computation |
| `causal_mask(seq_len)` | Upper-triangular attention mask |
| `sinusoidal_pe(seq_len, d_model)` | Positional encoding |
| `make_attention_weights(seq_len, n_heads)` | Generate demo attention weights |
| `make_token_sequence(words)` | Create demo token sequence |
| `value_to_color(v)` | Map float to color |
| `attention_palette(weights)` | Generate color palette from weights |
| `apply_dark_theme(scene)` | Set dark background theme |

### Tokenization (`manim_transformer.tokenization.token_box`)

| Class | Description |
|---|---|
| `TokenRow(tokens, token_ids)` | Row of token boxes |
| `TokenBox(text, token_id)` | Individual token visualization |
| `SpecialTokenBox(text)` | Special tokens ([CLS], [SEP], etc.) |
| `BPESequence` | Byte-pair encoding visualization |
| `VocabTable` | Vocabulary lookup table |
| `ContextWindowBar` | Context window indicator |

---

## 🎨 Color System

All colors are available in `manim_transformer.styles.colors`:

| Constant | Hex | Usage |
|---|---|---|
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
| `DIM_COLOR` | `#555555` | Dimmed elements |

---

## 📹 Example Videos

The `examples/` directory contains complete video scripts:

| # | File | Scene | Topic | Duration |
|---|---|---|---|---|
| 1 | `01_what_is_a_token.py` | `WhatIsAToken` | Tokenization basics | 3–4 min |
| 2 | `02_token_embeddings.py` | `TokenEmbeddings` | Embedding lookup | 4 min |
| 3 | `03_attention_mechanism.py` | `AttentionMechanism` | Q, K, V & softmax | 5–6 min |
| 4 | `04_multi_head_attention.py` | `MultiHeadAttention` | Multiple attention heads | 4 min |
| 5 | `05_full_forward_pass.py` | `FullForwardPass` | Complete forward pass | 5 min |

---

## 🚀 Video Production Workflow

1. **Plan**: Write your script outline and narration
2. **Import**: Choose components from `manim_transformer`
3. **Sketch**: Plan scene layout on paper
4. **Code**: Build scenes incrementally, preview with `-pql`
5. **Render**: Final render with `-pqh` or `-pqk`
6. **Voiceover**: Record audio to match animation timing
7. **Edit**: Combine in your video editor of choice

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

[Add your license here]

---

## 🙏 Acknowledgments

Built with [Manim Community Edition](https://www.manim.community/)

---

## 📬 Contact

[Your contact information or links]