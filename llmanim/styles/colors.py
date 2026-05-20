"""
manim_transformer/styles/colors.py
====================================
Central color palette for all Transformer/LLM animation assets.

Usage
-----
from manim_transformer.styles.colors import COLORS, get_color, component_color

Or import specific palettes:
from manim_transformer.styles.colors import (
    COMPONENT_COLORS,
    MATH_COLORS,
    TOKEN_PALETTE,
    UI_COLORS,
    HEATMAP_COLORS,
)
"""

from manim.utils.color import ManimColor


# ---------------------------------------------------------------------------
# RAW HEX PALETTE  (edit these to retheme the entire library)
# ---------------------------------------------------------------------------

# -- Transformer Components --------------------------------------------------
_COMPONENT_HEX = {
    "token":      "#4CAF50",   # green      — tokenization layer
    "embedding":  "#2196F3",   # blue       — embedding / lookup
    "attention":  "#9C27B0",   # purple     — multi-head attention
    "ffn":        "#FF9800",   # orange     — feed-forward network
    "norm":       "#00BCD4",   # cyan       — layer norm / RMS norm
    "output":     "#F44336",   # red        — logits / output head
    "residual":   "#8BC34A",   # light green— residual / skip connection
    "block":      "#607D8B",   # blue-grey  — full transformer block wrapper
    "encoder":    "#3F51B5",   # indigo     — encoder-specific
    "decoder":    "#E91E63",   # pink       — decoder-specific
    "cross_attn": "#FF5722",   # deep orange— cross-attention (enc–dec)
}

# -- QKV Math Elements -------------------------------------------------------
_MATH_HEX = {
    "query":         "#E91E63",   # pink
    "key":           "#3F51B5",   # indigo
    "value":         "#009688",   # teal
    "weight_matrix": "#795548",   # brown
    "bias":          "#9E9E9E",   # grey
    "softmax":       "#FF9800",   # orange
    "score":         "#FFC107",   # amber
    "logit":         "#F44336",   # red
}

# -- Per-token rotating palette (up to 8 tokens before repeating) ------------
_TOKEN_PALETTE_HEX = [
    "#42A5F5",   # blue
    "#66BB6A",   # green
    "#FFA726",   # orange
    "#AB47BC",   # purple
    "#26C6DA",   # cyan
    "#EC407A",   # pink
    "#FFCA28",   # yellow
    "#8D6E63",   # brown
]

# -- Special token colors ----------------------------------------------------
_SPECIAL_TOKEN_HEX = {
    "CLS":   "#FFD700",   # gold
    "SEP":   "#C0C0C0",   # silver
    "PAD":   "#555555",   # dark grey
    "MASK":  "#FF6B6B",   # coral
    "BOS":   "#B8F3B8",   # light green
    "EOS":   "#F3B8B8",   # light red
    "UNK":   "#F5A623",   # amber
}

# -- General UI / Scene colors -----------------------------------------------
_UI_HEX = {
    "background":      "#0D0D0D",   # near-black scene background
    "surface":         "#1A1A2E",   # card / panel background
    "surface_alt":     "#16213E",   # alternate panel
    "highlight":       "#FFEB3B",   # yellow highlight / emphasis
    "highlight_soft":  "#FFF9C4",   # soft yellow
    "arrow":           "#FFFFFF",   # default arrow / line
    "arrow_dim":       "#888888",   # de-emphasised arrow
    "dim":             "#555555",   # dimmed / inactive element
    "dim_soft":        "#333333",   # even softer dim
    "label":           "#EEEEEE",   # default label text
    "label_dim":       "#888888",   # secondary / small label
    "grid_line":       "#2A2A3E",   # grid / axis lines
    "border":          "#444466",   # default border
    "border_bright":   "#AAAACC",   # highlighted border
    "positive":        "#2196F3",   # positive value (matrix cell)
    "negative":        "#F44336",   # negative value (matrix cell)
    "zero":            "#263238",   # near-zero / empty cell
    "glow":            "#FFFFFF",   # glow / bloom colour
}

# -- Attention heatmap gradient (low → high attention) -----------------------
_HEATMAP_HEX = {
    "low":    "#0D0D0D",   # black      — 0.0 attention
    "mid_lo": "#1A237E",   # dark blue  — ~0.25
    "mid":    "#7B1FA2",   # purple     — ~0.5
    "mid_hi": "#E65100",   # dark orange— ~0.75
    "high":   "#FFEB3B",   # yellow     — 1.0 attention
}

# -- KV Cache / Memory -------------------------------------------------------
_MEMORY_HEX = {
    "cache_used":  "#4CAF50",   # green  — occupied cache slot
    "cache_free":  "#263238",   # dark   — empty cache slot
    "cache_new":   "#FFEB3B",   # yellow — newly added this step
    "cache_evict": "#F44336",   # red    — evicted slot
}

# -- Temperature / sampling --------------------------------------------------
_SAMPLING_HEX = {
    "temp_cold":   "#2196F3",   # blue   — low temperature (sharp)
    "temp_warm":   "#FF9800",   # orange — medium temperature
    "temp_hot":    "#F44336",   # red    — high temperature (flat)
    "top_k":       "#4CAF50",   # green  — kept token
    "top_k_cut":   "#555555",   # grey   — filtered-out token
    "selected":    "#FFEB3B",   # yellow — finally sampled token
}


# ---------------------------------------------------------------------------
# MANIM COLOR OBJECTS  (ManimColor wrappers — use these in your scenes)
# ---------------------------------------------------------------------------

def _to_manim(hex_dict: dict) -> dict:
    """Convert a ``{name: hex}`` dict into ``{name: ManimColor}``."""
    return {k: ManimColor(v) for k, v in hex_dict.items()}


COMPONENT_COLORS: dict[str, ManimColor] = _to_manim(_COMPONENT_HEX)
MATH_COLORS:      dict[str, ManimColor] = _to_manim(_MATH_HEX)
SPECIAL_TOKEN_COLORS: dict[str, ManimColor] = _to_manim(_SPECIAL_TOKEN_HEX)
UI_COLORS:        dict[str, ManimColor] = _to_manim(_UI_HEX)
HEATMAP_COLORS:   dict[str, ManimColor] = _to_manim(_HEATMAP_HEX)
MEMORY_COLORS:    dict[str, ManimColor] = _to_manim(_MEMORY_HEX)
SAMPLING_COLORS:  dict[str, ManimColor] = _to_manim(_SAMPLING_HEX)

TOKEN_PALETTE: list[ManimColor] = [ManimColor(h) for h in _TOKEN_PALETTE_HEX]


# ---------------------------------------------------------------------------
# UNIFIED FLAT DICT  (backwards-compat / quick access)
# ---------------------------------------------------------------------------

COLORS: dict[str, ManimColor] = {
    **COMPONENT_COLORS,
    **MATH_COLORS,
    **UI_COLORS,
}
"""
Flat merged dict with the most-used component + math + UI colors.
Keys from later dicts silently overwrite earlier ones — prefer the
typed sub-dicts (COMPONENT_COLORS, MATH_COLORS, …) when the key
might clash.
"""


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def get_color(name: str) -> ManimColor:
    """
    Look up a color by name across *all* palettes.

    Search order:
        COMPONENT_COLORS → MATH_COLORS → UI_COLORS →
        SPECIAL_TOKEN_COLORS → HEATMAP_COLORS →
        MEMORY_COLORS → SAMPLING_COLORS

    Raises ``KeyError`` with a helpful message if not found.

    Examples
    --------
    >>> get_color("attention")   # ManimColor("#9C27B0")
    >>> get_color("query")       # ManimColor("#E91E63")
    >>> get_color("highlight")   # ManimColor("#FFEB3B")
    """
    all_palettes = (
        COMPONENT_COLORS,
        MATH_COLORS,
        UI_COLORS,
        SPECIAL_TOKEN_COLORS,
        HEATMAP_COLORS,
        MEMORY_COLORS,
        SAMPLING_COLORS,
    )
    for palette in all_palettes:
        if name in palette:
            return palette[name]
    available = sorted({k for p in all_palettes for k in p})
    raise KeyError(
        f"Color '{name}' not found. Available keys:\n  {', '.join(available)}"
    )


def component_color(component: str) -> ManimColor:
    """
    Return the canonical color for a named Transformer component.

    Parameters
    ----------
    component : str
        One of: token, embedding, attention, ffn, norm, output,
        residual, block, encoder, decoder, cross_attn.

    Raises ``KeyError`` if the component is unknown.
    """
    if component not in COMPONENT_COLORS:
        raise KeyError(
            f"Unknown component '{component}'. "
            f"Valid options: {list(COMPONENT_COLORS)}"
        )
    return COMPONENT_COLORS[component]


def token_color(index: int) -> ManimColor:
    """
    Return a rotating per-token color from TOKEN_PALETTE.

    Parameters
    ----------
    index : int
        Token position in a sequence (wraps around automatically).

    Examples
    --------
    >>> token_color(0)   # blue
    >>> token_color(8)   # blue again (wraps)
    """
    return TOKEN_PALETTE[index % len(TOKEN_PALETTE)]


def special_token_color(token_type: str) -> ManimColor:
    """
    Return the color for a special token like [CLS], [SEP], [PAD], etc.

    Parameters
    ----------
    token_type : str
        One of: CLS, SEP, PAD, MASK, BOS, EOS, UNK.
        Lookup is case-insensitive.

    Falls back to ``UI_COLORS['dim']`` for unknown special tokens.
    """
    key = token_type.upper()
    return SPECIAL_TOKEN_COLORS.get(key, UI_COLORS["dim"])


def heatmap_color(attention_weight: float) -> str:
    """
    Map an attention weight in [0, 1] to an interpolated hex color string
    for use in attention heatmap cells.

    Returns a hex string (e.g. ``"#7B1FA2"``) so it can also be used
    directly as a Manim color or in matplotlib/PIL contexts.

    Uses a 5-stop gradient:
        0.00 → low
        0.25 → mid_lo
        0.50 → mid
        0.75 → mid_hi
        1.00 → high
    """
    stops = [
        (0.00, _HEATMAP_HEX["low"]),
        (0.25, _HEATMAP_HEX["mid_lo"]),
        (0.50, _HEATMAP_HEX["mid"]),
        (0.75, _HEATMAP_HEX["mid_hi"]),
        (1.00, _HEATMAP_HEX["high"]),
    ]

    t = max(0.0, min(1.0, attention_weight))

    # Find surrounding stops
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)
            return _lerp_hex(c0, c1, alpha)

    return _HEATMAP_HEX["high"]


def value_color(value: float, vmin: float = -1.0, vmax: float = 1.0) -> ManimColor:
    """
    Map a matrix/vector float value to a colour:
        negative → UI_COLORS['negative']  (red)
        near zero → UI_COLORS['zero']     (dark)
        positive  → UI_COLORS['positive'] (blue)

    Intensity scales with magnitude relative to ``vmin``/``vmax``.
    """
    # Normalise to [-1, 1]
    span = max(abs(vmin), abs(vmax), 1e-9)
    t = max(-1.0, min(1.0, value / span))

    if t >= 0:
        return ManimColor(_lerp_hex(_UI_HEX["zero"], _UI_HEX["positive"], t))
    else:
        return ManimColor(_lerp_hex(_UI_HEX["zero"], _UI_HEX["negative"], -t))


# ---------------------------------------------------------------------------
# INTERNAL UTILITIES
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Parse ``'#RRGGBB'`` → ``(R, G, B)`` ints 0-255."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lerp_hex(hex_a: str, hex_b: str, t: float) -> str:
    """Linear-interpolate between two hex colours; return hex string."""
    r0, g0, b0 = _hex_to_rgb(hex_a)
    r1, g1, b1 = _hex_to_rgb(hex_b)
    r = int(r0 + (r1 - r0) * t)
    g = int(g0 + (g1 - g0) * t)
    b = int(b0 + (b1 - b0) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


# ---------------------------------------------------------------------------
# QUICK REFERENCE  (run ``python -m manim_transformer.styles.colors``)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  manim_transformer  —  Color Palette Reference")
    print("=" * 60)

    sections = {
        "Component Colors":     _COMPONENT_HEX,
        "Math / QKV Colors":    _MATH_HEX,
        "Special Token Colors": _SPECIAL_TOKEN_HEX,
        "UI / Scene Colors":    _UI_HEX,
        "Heatmap Gradient":     _HEATMAP_HEX,
        "Memory / KV Cache":    _MEMORY_HEX,
        "Sampling Colors":      _SAMPLING_HEX,
    }

    for section, palette in sections.items():
        print(f"\n  {section}")
        print("  " + "-" * (len(section) + 2))
        for name, hex_val in palette.items():
            print(f"    {name:<20} {hex_val}")

    print("\n  Token Palette  (index-based, wrapping)")
    print("  " + "-" * 38)
    for i, hex_val in enumerate(_TOKEN_PALETTE_HEX):
        print(f"    index {i}              {hex_val}")

    print("\n  Heatmap interpolation test")
    print("  " + "-" * 38)
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"    weight={w:.2f}  →  {heatmap_color(w)}")