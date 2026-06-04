# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-01-01

### Added
- Initial release
- `MatrixBox`, `VectorBar`, `TokenBox`, `LabeledBlock` base shapes
- `TokenRow`, `SpecialTokenBox`, `BPESequence`, `VocabTable` tokenization components
- Attention mechanism components: QKV projection, attention scores, attention output, multi-head
- Feed-forward network (`ffn.py`) and layer normalization components
- Embedding lookup and sinusoidal positional encoding
- Output logits and sampling visualization
- Full transformer block and `full_model.py`
- Reusable animation functions: `flow_through`, `attention_flow`, `data_flow_arrow`, `highlight_sequence`
- Math utilities: `softmax`, `scaled_dot_product`, `causal_mask`, `sinusoidal_pe`
- Dark theme support via `apply_dark_theme`
- Color system in `llmanim.styles.colors`
- 5 example video scripts covering tokenization → full forward pass