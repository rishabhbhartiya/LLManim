"""
manim_transformer/base/utils.py
================================
Pure utility helpers for the manim_transformer library.
No Manim scene or animation logic here — only:
  • Math helpers   (softmax, scaled dot-product, positional encoding, etc.)
  • Layout helpers (auto-arrange, grid positions, bounding-box arithmetic)
  • Color helpers  (value-to-color mapping, interpolation, heatmap palette)
  • Text helpers   (number formatting, LaTeX builders, truncation)
  • Timing helpers (run-time calculators, lag schedules)
  • Data helpers   (fake token sequences, random weight matrices, sample logits)
  • Logging        (scene step printer, shape inspector)

Usage:
    from manim_transformer.base.utils import (
        softmax, scaled_dot_product,
        value_to_color, heatmap_color,
        grid_positions, auto_arrange_row,
        fmt_num, latex_matrix, truncate_vector,
        step_log, inspect_shape,
        make_token_sequence, make_weight_matrix, make_sample_logits,
    )
"""

from __future__ import annotations

import math
import textwrap
import time
from typing import Sequence

import numpy as np


# ─────────────────────────────────────────────────────────────
# COLOR PALETTE  (keep in sync with shapes.py)
# ─────────────────────────────────────────────────────────────

_TOKEN_COLOR     = "#4CAF50"
_EMBEDDING_COLOR = "#2196F3"
_ATTENTION_COLOR = "#9C27B0"
_FFN_COLOR       = "#FF9800"
_NORM_COLOR      = "#00BCD4"
_OUTPUT_COLOR    = "#F44336"
_QUERY_COLOR     = "#E91E63"
_KEY_COLOR       = "#3F51B5"
_VALUE_COLOR     = "#009688"
_HIGHLIGHT_COLOR = "#FFEB3B"
_DIM_COLOR       = "#555555"
_POSITIVE_COLOR  = "#2196F3"
_NEGATIVE_COLOR  = "#F44336"


# ═══════════════════════════════════════════════════════════
# ➤  SECTION 1 — MATH HELPERS
# ═══════════════════════════════════════════════════════════

def softmax(x: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax with temperature scaling.

    Parameters
    ----------
    x           : np.ndarray – raw logits (any shape)
    temperature : float      – divide logits by T before softmax
                               T→0 → one-hot,  T→∞ → uniform
    axis        : int        – axis to apply softmax over

    Returns
    -------
    np.ndarray same shape as x, values in (0, 1) summing to 1 along `axis`

    Example
    -------
    probs = softmax(np.array([2.0, 1.0, 0.1]))
    probs = softmax(score_matrix, temperature=0.5, axis=-1)
    """
    x = np.array(x, dtype=float)
    x_shifted = (x - x.max(axis=axis, keepdims=True)) / temperature
    e = np.exp(x_shifted)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full scaled dot-product attention.

    Parameters
    ----------
    Q    : (seq_q, d_k)
    K    : (seq_k, d_k)
    V    : (seq_k, d_v)
    mask : (seq_q, seq_k) boolean — True means MASK (set to -inf)
    temperature : float  – extra temperature on top of √d_k scaling

    Returns
    -------
    (output, attention_weights)
    output            : (seq_q, d_v)
    attention_weights : (seq_q, seq_k)  — after softmax

    Example
    -------
    Q, K, V = np.random.randn(6, 64), np.random.randn(6, 64), np.random.randn(6, 64)
    out, weights = scaled_dot_product(Q, K, V)
    """
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / (math.sqrt(d_k) * temperature)   # (seq_q, seq_k)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    weights = softmax(scores, axis=-1)
    output  = weights @ V
    return output, weights


def causal_mask(seq_len: int) -> np.ndarray:
    """
    Upper-triangular boolean mask for causal (GPT-style) attention.
    True  → position is MASKED (future token, set to -inf).
    False → position is visible.

    Parameters
    ----------
    seq_len : int

    Returns
    -------
    (seq_len, seq_len) bool array

    Example
    -------
    mask = causal_mask(6)
    scores_masked = np.where(mask, -1e9, scores)
    """
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


def sinusoidal_pe(seq_len: int, d_model: int) -> np.ndarray:
    """
    Classic sinusoidal positional encoding from "Attention Is All You Need".

    Parameters
    ----------
    seq_len : int
    d_model : int

    Returns
    -------
    (seq_len, d_model) float array

    Example
    -------
    pe = sinusoidal_pe(seq_len=20, d_model=512)
    """
    pe  = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]                     # (seq_len, 1)
    i   = np.arange(0, d_model, 2)                        # even indices
    div = np.exp(i * (-math.log(10000.0) / d_model))      # (d_model/2,)

    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Layer normalisation (no learnable γ/β — pure normalisation).

    Parameters
    ----------
    x   : np.ndarray – shape (*, d_model)
    eps : float      – numerical stability

    Returns
    -------
    np.ndarray same shape

    Example
    -------
    normed = layer_norm(hidden_states)
    """
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1,  keepdims=True)
    return (x - mean) / (std + eps)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU activation (used in GPT-2/3, BERT).
    Approximation: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    """
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish activation (used in LLaMA)."""
    return x / (1.0 + np.exp(-x))


def entropy(probs: np.ndarray) -> float:
    """
    Shannon entropy of a probability distribution.
    High entropy → flat / uncertain.  Low entropy → peaked / confident.

    Parameters
    ----------
    probs : np.ndarray – 1-D probabilities summing to 1

    Returns
    -------
    float  (nats)
    """
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """
    Zero out all logits except the top-k.
    Returns a copy — original unchanged.
    """
    logits = logits.copy()
    threshold = np.sort(logits)[-k]
    logits[logits < threshold] = -np.inf
    return logits


def top_p_filter(logits: np.ndarray, p: float = 0.9) -> np.ndarray:
    """
    Nucleus (top-p) filtering.
    Keep the smallest set of tokens whose cumulative probability ≥ p.
    Returns a copy with masked logits.
    """
    logits  = logits.copy()
    probs   = softmax(logits)
    sorted_idx   = np.argsort(probs)[::-1]
    cumsum_probs = np.cumsum(probs[sorted_idx])
    cutoff_idx   = np.searchsorted(cumsum_probs, p)
    masked_idx   = sorted_idx[cutoff_idx + 1:]
    logits[masked_idx] = -np.inf
    return logits


# ═══════════════════════════════════════════════════════════
# ➤  SECTION 2 — LAYOUT HELPERS
# ═══════════════════════════════════════════════════════════

def grid_positions(
    rows: int,
    cols: int,
    cell_w: float = 1.0,
    cell_h: float = 1.0,
    center: Sequence[float] = (0, 0, 0),
) -> list[np.ndarray]:
    """
    Return a flat list of (x, y, 0) positions for a rows×cols grid,
    centred at `center`.

    Parameters
    ----------
    rows, cols : int
    cell_w     : float – horizontal spacing
    cell_h     : float – vertical spacing
    center     : (x, y, z)

    Returns
    -------
    list of np.ndarray, length rows*cols, row-major order

    Example
    -------
    pos = grid_positions(4, 6, cell_w=0.5, cell_h=0.5)
    for mob, p in zip(my_mobjects, pos):
        mob.move_to(p)
    """
    cx, cy, cz = center
    positions  = []
    x_start    = cx - (cols - 1) * cell_w / 2
    y_start    = cy + (rows - 1) * cell_h / 2

    for r in range(rows):
        for c in range(cols):
            x = x_start + c * cell_w
            y = y_start - r * cell_h
            positions.append(np.array([x, y, cz]))
    return positions


def auto_arrange_row(
    mobjects: list,
    buff: float = 0.2,
    center: Sequence[float] = (0, 0, 0),
) -> list[np.ndarray]:
    """
    Return positions that place `mobjects` evenly in a horizontal row,
    respecting each object's width.

    Parameters
    ----------
    mobjects : list of Manim Mobjects (need .width attribute)
    buff     : float – gap between objects
    center   : (x, y, z) – centre of the row

    Returns
    -------
    list of np.ndarray positions (same length as mobjects)

    Example
    -------
    positions = auto_arrange_row(token_boxes, buff=0.15)
    for box, pos in zip(token_boxes, positions):
        box.move_to(pos)
    """
    widths    = [m.width for m in mobjects]
    total_w   = sum(widths) + buff * (len(mobjects) - 1)
    x_start   = center[0] - total_w / 2
    positions = []
    cursor    = x_start

    for w in widths:
        positions.append(np.array([cursor + w / 2, center[1], center[2]]))
        cursor += w + buff
    return positions


def stack_positions(
    mobjects: list,
    buff: float = 0.15,
    direction: Sequence[float] = (0, 1, 0),
    center: Sequence[float] = (0, 0, 0),
) -> list[np.ndarray]:
    """
    Return positions that stack `mobjects` along `direction`.

    Parameters
    ----------
    mobjects  : list
    buff      : float
    direction : unit vector (UP=+y, DOWN=-y, RIGHT=+x, LEFT=-x)
    center    : centre of the stack

    Returns
    -------
    list of np.ndarray positions

    Example
    -------
    pos = stack_positions(layer_blocks, buff=0.2, direction=(0, 1, 0))
    """
    d = np.array(direction, dtype=float)
    d_norm = d / (np.linalg.norm(d) + 1e-9)

    heights = []
    for m in mobjects:
        # project bounding-box size onto direction
        size = abs(m.width * d_norm[0]) + abs(m.height * d_norm[1])
        heights.append(size)

    total = sum(heights) + buff * (len(mobjects) - 1)
    start = np.array(center) - d_norm * total / 2

    positions = []
    cursor    = 0.0
    for h in heights:
        positions.append(start + d_norm * (cursor + h / 2))
        cursor += h + buff
    return positions


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the midpoint between two 3-D points."""
    return (np.array(a) + np.array(b)) / 2


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two points.  t=0 → a,  t=1 → b."""
    return np.array(a) + t * (np.array(b) - np.array(a))


def bounding_box(mobjects: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (min_corner, max_corner) of the combined bounding box
    of a list of Manim Mobjects.
    """
    mins, maxs = [], []
    for m in mobjects:
        mins.append(m.get_corner([-1, -1, 0]))
        maxs.append(m.get_corner([ 1,  1, 0]))
    return np.min(mins, axis=0), np.max(maxs, axis=0)


# ═══════════════════════════════════════════════════════════
# ➤  SECTION 3 — COLOR HELPERS
# ═══════════════════════════════════════════════════════════

def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """
    Convert "#RRGGBB" → (r, g, b) floats in [0, 1].

    Example
    -------
    r, g, b = hex_to_rgb("#E91E63")
    """
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """
    Convert (r, g, b) floats in [0, 1] → "#RRGGBB".

    Example
    -------
    color = rgb_to_hex(0.9, 0.12, 0.39)   # "#E61F63"
    """
    return "#{:02X}{:02X}{:02X}".format(
        int(r * 255), int(g * 255), int(b * 255)
    )


def lerp_color(color_a: str, color_b: str, t: float) -> str:
    """
    Linearly interpolate between two hex colors.
    t=0 → color_a,  t=1 → color_b.

    Example
    -------
    mid = lerp_color("#2196F3", "#F44336", 0.5)  # purple-ish
    """
    ra, ga, ba = hex_to_rgb(color_a)
    rb, gb, bb = hex_to_rgb(color_b)
    t = float(np.clip(t, 0, 1))
    return rgb_to_hex(ra + t*(rb-ra), ga + t*(gb-ga), ba + t*(bb-ba))


def value_to_color(
    value: float,
    v_min: float = -1.0,
    v_max: float =  1.0,
    neg_color: str = _NEGATIVE_COLOR,
    zero_color: str = "#1E1E2E",
    pos_color: str  = _POSITIVE_COLOR,
) -> str:
    """
    Map a scalar value to a color on a diverging scale.
    Negative → neg_color,  zero → zero_color,  positive → pos_color.

    Parameters
    ----------
    value           : float
    v_min, v_max    : float – clipping range
    neg_color       : str   – hex color for minimum
    zero_color      : str   – hex color for zero
    pos_color       : str   – hex color for maximum

    Returns
    -------
    str hex color

    Example
    -------
    color = value_to_color(0.75)    # mostly blue
    color = value_to_color(-0.5)    # pinkish red
    """
    v = float(np.clip(value, v_min, v_max))
    if v >= 0:
        t = v / (v_max if v_max != 0 else 1)
        return lerp_color(zero_color, pos_color, t)
    else:
        t = v / (v_min if v_min != 0 else -1)
        return lerp_color(zero_color, neg_color, t)


def heatmap_color(
    value: float,
    v_min: float = 0.0,
    v_max: float = 1.0,
    cold: str = "#1E1E2E",
    hot:  str = _HIGHLIGHT_COLOR,
) -> str:
    """
    Map value → heat color (dark=cold, bright=hot).
    Used for attention heatmaps.

    Parameters
    ----------
    value       : float
    v_min, v_max: float
    cold        : str – color at minimum
    hot         : str – color at maximum

    Returns
    -------
    str hex color

    Example
    -------
    # Color an attention weight
    for i, w in enumerate(attention_weights):
        cell.set_fill(heatmap_color(w), opacity=0.9)
    """
    t = float(np.clip((value - v_min) / (v_max - v_min + 1e-9), 0, 1))
    return lerp_color(cold, hot, t)


def opacity_for_value(value: float, v_min: float = 0.0, v_max: float = 1.0,
                      min_opacity: float = 0.05, max_opacity: float = 0.95) -> float:
    """
    Map a value to an opacity level.
    Great for showing attention weights as transparency.

    Example
    -------
    cell.set_fill(ATTENTION_COLOR, opacity=opacity_for_value(weight))
    """
    t = float(np.clip((value - v_min) / (v_max - v_min + 1e-9), 0, 1))
    return min_opacity + t * (max_opacity - min_opacity)


def attention_palette(
    weights: np.ndarray,
    base_color: str = _ATTENTION_COLOR,
) -> list[str]:
    """
    Convert a 1-D attention weight array into a list of colors,
    from dark (low attention) to bright (high attention).

    Parameters
    ----------
    weights    : np.ndarray  – 1-D array (will be normalised)
    base_color : str         – the "hot" color

    Returns
    -------
    list of hex color strings, same length as weights

    Example
    -------
    colors = attention_palette(np.array([0.1, 0.6, 0.2, 0.1]))
    for cell, c in zip(cells, colors):
        cell.set_fill(c, opacity=0.9)
    """
    w = np.array(weights, dtype=float)
    w = (w - w.min()) / (w.max() - w.min() + 1e-9)
    return [heatmap_color(v, cold="#1E1E2E", hot=base_color) for v in w]


# ═══════════════════════════════════════════════════════════
# ➤  SECTION 4 — TEXT & LATEX HELPERS
# ═══════════════════════════════════════════════════════════

def fmt_num(value: float, decimals: int = 2, sci: bool = False) -> str:
    """
    Format a float for display inside an animation.

    Parameters
    ----------
    value    : float
    decimals : int   – decimal places
    sci      : bool  – use scientific notation for very small/large numbers

    Returns
    -------
    str

    Example
    -------
    fmt_num(0.000123, sci=True)   # "1.23e-04"
    fmt_num(3.14159, decimals=3)  # "3.142"
    """
    if sci or (abs(value) < 0.001 and value != 0):
        return f"{value:.{decimals}e}"
    return f"{value:.{decimals}f}"


def truncate_vector(values: np.ndarray, show: int = 4) -> str:
    """
    Produce a compact string for a long vector.
    Shows first `show` values then "...".

    Parameters
    ----------
    values : np.ndarray – 1-D
    show   : int        – how many values to display

    Returns
    -------
    str like "[0.12, -0.34, 0.89, 0.01, ...]"

    Example
    -------
    truncate_vector(np.random.randn(512), show=4)
    # "[0.23, -0.11, 0.88, -0.44, ...]"
    """
    nums = [fmt_num(float(v)) for v in values[:show]]
    suffix = ", ...]" if len(values) > show else "]"
    return "[" + ", ".join(nums) + suffix


def latex_matrix(
    values: np.ndarray,
    max_rows: int = 4,
    max_cols: int = 4,
    decimals: int = 1,
) -> str:
    """
    Build a LaTeX bmatrix string from a 2-D numpy array.
    Truncates to max_rows × max_cols with "\\vdots" / "\\cdots".

    Parameters
    ----------
    values   : np.ndarray – 2-D
    max_rows : int
    max_cols : int
    decimals : int

    Returns
    -------
    str  (pass directly to MathTex)

    Example
    -------
    tex = latex_matrix(np.random.randn(3, 3))
    MathTex(tex)
    """
    r, c = values.shape
    rows_to_show = min(r, max_rows)
    cols_to_show = min(c, max_cols)
    show_vdots   = r > max_rows
    show_cdots   = c > max_cols

    lines = []
    for i in range(rows_to_show):
        row_vals = [f"{values[i, j]:.{decimals}f}" for j in range(cols_to_show)]
        if show_cdots:
            row_vals.append(r"\cdots")
        lines.append(" & ".join(row_vals))
    if show_vdots:
        vdots_row = " & ".join([r"\vdots"] * (cols_to_show + int(show_cdots)))
        lines.append(vdots_row)

    inner = r" \\ ".join(lines)
    return r"\begin{bmatrix}" + inner + r"\end{bmatrix}"


def latex_vector(
    values: np.ndarray,
    max_show: int = 5,
    decimals: int = 2,
    label: str = "",
) -> str:
    """
    Build a LaTeX column vector string.

    Parameters
    ----------
    values   : np.ndarray – 1-D
    max_show : int        – max elements before truncation
    decimals : int
    label    : str        – prepend "label = " if given

    Returns
    -------
    str LaTeX

    Example
    -------
    tex = latex_vector(query_vec, label="q_1")
    """
    to_show = values[:max_show]
    rows    = [f"{v:.{decimals}f}" for v in to_show]
    if len(values) > max_show:
        rows.append(r"\vdots")
    inner = r" \\ ".join(rows)
    vec   = r"\begin{bmatrix}" + inner + r"\end{bmatrix}"
    return (label + " = " + vec) if label else vec


def shape_label(arr: np.ndarray) -> str:
    """
    Return a human-readable shape string.

    Example
    -------
    shape_label(np.zeros((6, 512)))  # "(6 × 512)"
    """
    return "(" + " × ".join(str(s) for s in arr.shape) + ")"


def wrap_text(text: str, width: int = 40) -> str:
    """
    Wrap long text to `width` chars per line.
    Useful for subtitles and explanatory captions.
    """
    return "\n".join(textwrap.wrap(text, width=width))


def ordinal(n: int) -> str:
    """
    Return ordinal string: 1→"1st", 2→"2nd", 3→"3rd", 4→"4th" …

    Example
    -------
    ordinal(3)  # "3rd"
    """
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ═══════════════════════════════════════════════════════════
# ➤  SECTION 5 — TIMING HELPERS
# ═══════════════════════════════════════════════════════════

def lag_schedule(
    n: int,
    style: str = "linear",
    total_time: float = 2.0,
) -> list[float]:
    """
    Generate a list of lag values for AnimationGroup.

    Parameters
    ----------
    n          : int   – number of items
    style      : str   – "linear" | "ease_in" | "ease_out" | "random"
    total_time : float – desired total animation duration

    Returns
    -------
    list of floats of length n, summing approximately to total_time

    Example
    -------
    lags = lag_schedule(10, style="ease_in", total_time=2.0)
    AnimationGroup(*anims, lag_ratio=lags[0])   # or loop with individual waits
    """
    if n == 0:
        return []

    if style == "linear":
        weights = np.ones(n)
    elif style == "ease_in":
        weights = np.linspace(0.3, 1.0, n)
    elif style == "ease_out":
        weights = np.linspace(1.0, 0.3, n)
    elif style == "random":
        weights = np.random.uniform(0.5, 1.5, n)
    else:
        weights = np.ones(n)

    weights = weights / weights.sum()
    return (weights * total_time).tolist()


def run_time_for_seq(seq_len: int, base: float = 0.25, max_rt: float = 4.0) -> float:
    """
    Suggest a total run_time for animating a sequence of `seq_len` items.
    Scales sub-linearly so long sequences don't take forever.

    Parameters
    ----------
    seq_len : int
    base    : float – seconds per item (short sequences)
    max_rt  : float – cap

    Returns
    -------
    float (seconds)

    Example
    -------
    rt = run_time_for_seq(len(tokens))
    self.play(token_stream_anim(boxes, positions), run_time=rt)
    """
    rt = base * math.sqrt(seq_len)
    return min(rt, max_rt)


def per_item_run_time(total: float, n: int, min_rt: float = 0.05) -> float:
    """
    Divide a total animation time equally among n items.

    Example
    -------
    rt = per_item_run_time(total=2.0, n=len(tokens))
    """
    return max(total / max(n, 1), min_rt)


# ═══════════════════════════════════════════════════════════
# ➤  SECTION 6 — DATA HELPERS  (generate realistic fake data)
# ═══════════════════════════════════════════════════════════

# A small fixed vocabulary for demos
DEMO_VOCAB = {
    "the": 0, "a": 1, "cat": 2, "sat": 3, "on": 4,
    "mat": 5, "transformer": 6, "attention": 7, "is": 8,
    "all": 9, "you": 10, "need": 11, "[CLS]": 12, "[SEP]": 13,
    "[PAD]": 14, "model": 15, "language": 16, "token": 17,
    "embedding": 18, "query": 19, "key": 20, "value": 21,
}
DEMO_VOCAB_INV = {v: k for k, v in DEMO_VOCAB.items()}


def make_token_sequence(
    words: list[str] | None = None,
    add_special: bool = True,
) -> tuple[list[str], list[int]]:
    """
    Create a demo token sequence with IDs.

    Parameters
    ----------
    words        : list[str] | None  – if None uses a default sentence
    add_special  : bool              – prepend [CLS], append [SEP]

    Returns
    -------
    (tokens, ids)  – both lists of same length

    Example
    -------
    tokens, ids = make_token_sequence(["the", "cat", "sat"])
    # tokens = ["[CLS]", "the", "cat", "sat", "[SEP]"]
    # ids    = [12, 0, 2, 3, 13]
    """
    if words is None:
        words = ["the", "transformer", "is", "all", "you", "need"]

    if add_special:
        words = ["[CLS]"] + list(words) + ["[SEP]"]

    ids = [DEMO_VOCAB.get(w, len(DEMO_VOCAB)) for w in words]
    return words, ids


def make_weight_matrix(
    rows: int,
    cols: int,
    seed: int | None = None,
    scale: float = 0.1,
) -> np.ndarray:
    """
    Generate a realistic-looking weight matrix (Xavier-like scale).

    Parameters
    ----------
    rows, cols : int
    seed       : int | None – for reproducibility
    scale      : float      – std deviation multiplier

    Returns
    -------
    np.ndarray shape (rows, cols)

    Example
    -------
    W_q = make_weight_matrix(512, 64, seed=42)
    """
    rng = np.random.default_rng(seed)
    std = scale / math.sqrt(rows + cols)
    return rng.normal(0.0, std, (rows, cols))


def make_attention_weights(
    seq_len: int,
    n_heads: int = 1,
    causal: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate plausible attention weight matrices for demo scenes.

    Parameters
    ----------
    seq_len : int
    n_heads : int  – number of attention heads
    causal  : bool – apply causal mask
    seed    : int | None

    Returns
    -------
    np.ndarray shape (n_heads, seq_len, seq_len)
    Each row sums to 1.

    Example
    -------
    weights = make_attention_weights(seq_len=6, n_heads=4, causal=True)
    head_0  = weights[0]   # (6, 6)
    """
    rng    = np.random.default_rng(seed)
    logits = rng.normal(0, 1, (n_heads, seq_len, seq_len))

    if causal:
        mask = causal_mask(seq_len)
        logits = np.where(mask[None, :, :], -1e9, logits)

    # softmax over last axis
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def make_sample_logits(
    vocab_size: int = 10,
    top_token: str = "the",
    seed: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Generate sample logits for an output distribution demo.
    One token gets a clearly higher logit.

    Parameters
    ----------
    vocab_size : int
    top_token  : str – which token should dominate
    seed       : int | None

    Returns
    -------
    (logits, token_labels)

    Example
    -------
    logits, labels = make_sample_logits(vocab_size=8, top_token="cat")
    sc = SoftmaxCurve(logits=logits, labels=labels)
    """
    rng    = np.random.default_rng(seed)
    logits = rng.uniform(-1.5, 1.0, vocab_size)

    # give top_token the highest logit
    top_idx = DEMO_VOCAB.get(top_token, 0) % vocab_size
    logits[top_idx] = rng.uniform(2.5, 3.5)

    labels = [DEMO_VOCAB_INV.get(i, f"tok{i}") for i in range(vocab_size)]
    return logits, labels


def make_embedding_matrix(
    vocab_size: int = 20,
    d_model: int = 16,
    seed: int | None = None,
) -> np.ndarray:
    """
    Create a small demo embedding matrix.

    Returns
    -------
    np.ndarray shape (vocab_size, d_model)

    Example
    -------
    E = make_embedding_matrix(vocab_size=20, d_model=16)
    token_vec = E[token_id]
    """
    rng = np.random.default_rng(seed)
    E   = rng.normal(0, 0.1, (vocab_size, d_model))
    # normalise rows
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return E / (norms + 1e-9)


# ═══════════════════════════════════════════════════════════
# ➤  SECTION 7 — LOGGING & INSPECTION
# ═══════════════════════════════════════════════════════════

_STEP_COUNTER: dict[str, int] = {}


def step_log(message: str, scene_name: str = "scene", color: bool = True) -> None:
    """
    Print a numbered step log to stdout while a scene is constructing.
    Useful for debugging long scenes.

    Parameters
    ----------
    message    : str
    scene_name : str – groups step counters per scene
    color      : bool – ANSI color output

    Example
    -------
    step_log("Building attention matrix",   scene_name="AttentionScene")
    step_log("Playing softmax animation",   scene_name="AttentionScene")
    """
    _STEP_COUNTER[scene_name] = _STEP_COUNTER.get(scene_name, 0) + 1
    step = _STEP_COUNTER[scene_name]

    ts  = time.strftime("%H:%M:%S")
    msg = f"[{ts}] [{scene_name}] Step {step:02d}: {message}"

    if color:
        CYAN, RESET = "\033[96m", "\033[0m"
        print(f"{CYAN}{msg}{RESET}")
    else:
        print(msg)


def inspect_shape(arr: np.ndarray, name: str = "array") -> str:
    """
    Print and return a formatted description of an array.

    Parameters
    ----------
    arr  : np.ndarray
    name : str

    Returns
    -------
    str description

    Example
    -------
    inspect_shape(attention_weights, "attn_weights")
    # attn_weights  shape=(4, 6, 6)  dtype=float64  min=-0.03  max=0.97
    """
    info = (
        f"{name:<20} shape={arr.shape!s:<20} dtype={arr.dtype!s:<12} "
        f"min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}"
    )
    print(info)
    return info


def validate_attention_weights(weights: np.ndarray, tol: float = 1e-4) -> bool:
    """
    Check that attention weights sum to 1 along the last axis.
    Raises ValueError if they don't.

    Parameters
    ----------
    weights : np.ndarray – (..., seq_len)
    tol     : float      – tolerance

    Returns
    -------
    True if valid

    Example
    -------
    validate_attention_weights(make_attention_weights(6))
    """
    row_sums = weights.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        raise ValueError(
            f"Attention weights do not sum to 1. "
            f"Got row sums: min={row_sums.min():.4f}, max={row_sums.max():.4f}"
        )
    return True


def assert_shape(arr: np.ndarray, expected: tuple, name: str = "array") -> None:
    """
    Assert an array has the expected shape; raise with clear message if not.

    Example
    -------
    assert_shape(Q, (seq_len, d_k), "Q")
    """
    if arr.shape != expected:
        raise AssertionError(
            f"{name}: expected shape {expected}, got {arr.shape}"
        )


# ═══════════════════════════════════════════════════════════
# ➤  SECTION 8 — SCENE CONFIG PRESETS
# ═══════════════════════════════════════════════════════════

def apply_dark_theme(scene) -> None:
    """
    Set background color to the library's dark theme.
    Call at the top of construct().

    Example
    -------
    def construct(self):
        apply_dark_theme(self)
        ...
    """
    scene.camera.background_color = "#1E1E2E"


QUALITY_PRESETS = {
    # key: (pixel_height, pixel_width, frame_rate)
    "preview":    (480,  854, 15),
    "medium":     (720, 1280, 30),
    "high":       (1080, 1920, 30),
    "4k":         (2160, 3840, 60),
}


def print_quality_hint(preset: str = "preview") -> None:
    """
    Print the manim CLI flag to render at a given quality preset.

    Example
    -------
    print_quality_hint("high")
    # → manim -pqh your_scene.py YourScene
    """
    flags = {"preview": "-pql", "medium": "-pqm", "high": "-pqh", "4k": "-pqk"}
    flag  = flags.get(preset, "-pql")
    print(f"Render hint:  manim {flag} your_scene.py YourScene")


# ─────────────────────────────────────────────────────────────
# Quick self-test  (python -m manim_transformer.base.utils)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("─── Math helpers ───────────────────────────────")
    probs = softmax(np.array([2.0, 1.0, 0.1, -0.5]))
    print("softmax:", np.round(probs, 3), "  sum:", round(probs.sum(), 4))

    Q = np.random.randn(4, 8)
    K = np.random.randn(4, 8)
    V = np.random.randn(4, 16)
    out, w = scaled_dot_product(Q, K, V, mask=causal_mask(4))
    print("scaled_dot_product output shape:", out.shape)
    validate_attention_weights(w)
    print("attention weights valid ✓")

    pe = sinusoidal_pe(seq_len=10, d_model=16)
    print("sinusoidal_pe shape:", pe.shape)

    print("\n─── Color helpers ──────────────────────────────")
    print("value_to_color(0.8):", value_to_color(0.8))
    print("value_to_color(-0.5):", value_to_color(-0.5))
    print("heatmap_color(0.7):", heatmap_color(0.7))
    pal = attention_palette(w[0])
    print("attention_palette (first 4):", pal[:4])

    print("\n─── Text helpers ───────────────────────────────")
    vec = np.random.randn(512)
    print("truncate_vector:", truncate_vector(vec))
    mat = np.random.randn(6, 6)
    print("latex_matrix (snippet):", latex_matrix(mat)[:60], "...")
    print("shape_label:", shape_label(mat))

    print("\n─── Data helpers ───────────────────────────────")
    tokens, ids = make_token_sequence()
    print("tokens:", tokens)
    print("ids:   ", ids)

    logits, labels = make_sample_logits(vocab_size=8, top_token="cat")
    print("sample logits:", np.round(logits, 2))
    print("labels:       ", labels)

    W = make_weight_matrix(64, 16, seed=0)
    inspect_shape(W, "W_q")

    weights = make_attention_weights(seq_len=6, n_heads=2, causal=True)
    inspect_shape(weights, "attn_weights")

    print("\n─── Timing helpers ─────────────────────────────")
    print("lag_schedule(5, ease_in):", [round(l, 3) for l in lag_schedule(5, "ease_in")])
    print("run_time_for_seq(20):    ", run_time_for_seq(20))

    print("\n─── Logging ────────────────────────────────────")
    step_log("Building token boxes",   scene_name="DemoScene")
    step_log("Playing attention anim", scene_name="DemoScene")

    print_quality_hint("high")
    print("\nAll utils tests passed ✓")