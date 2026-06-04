"""
llmanim/styles/constants.py
=======================================
Scene-wide constants for every Transformer/LLM animation asset.

All numeric "magic numbers" live here so changing one value
re-themes the whole library consistently.

Usage
-----
from llmanim.styles.constants import (
    # pick what you need
    SCENE, TIMING, LAYOUT, MATRIX, TOKEN,
    ATTENTION, EMBEDDING, FFN, SAMPLING,
    MODEL, Z_INDEX,
)
"""

from types import SimpleNamespace


# ---------------------------------------------------------------------------
# SCENE  — canvas / camera settings
# ---------------------------------------------------------------------------

SCENE = SimpleNamespace(
    # Manim default frame is 14.22 × 8.0 units (16:9 at FRAME_HEIGHT=8)
    WIDTH           = 14.222,   # frame width  in Manim units
    HEIGHT          = 8.0,      # frame height in Manim units
    MARGIN          = 0.5,      # safe-zone margin from any edge

    # Pixel resolution presets (pass to --resolution flag)
    RES_LOW         = (854,  480),    # quick preview
    RES_MEDIUM      = (1280, 720),    # draft render
    RES_HIGH        = (1920, 1080),   # final HD
    RES_4K          = (3840, 2160),   # final 4K

    # Camera
    BACKGROUND_HEX  = "#0D0D0D",      # must match UI_COLORS["background"]
    FRAME_RATE      = 60,             # fps for final renders
    FRAME_RATE_PREVIEW = 15,          # fps for quick previews
)


# ---------------------------------------------------------------------------
# TIMING  — all durations in seconds
# ---------------------------------------------------------------------------

TIMING = SimpleNamespace(
    # Standard animation beats
    INSTANT         = 0.0,    # no animation — snap
    FAST            = 0.3,    # quick micro-interaction
    NORMAL          = 0.6,    # default transition
    SLOW            = 1.2,    # deliberate / emphasis
    VERY_SLOW       = 2.0,    # dramatic reveal

    # Specific animation durations
    TOKEN_APPEAR    = 0.25,   # single TokenBox fades in
    TOKEN_SPLIT     = 0.4,    # text → token boxes split
    EMBED_LOOKUP    = 0.5,    # row extraction from EmbeddingMatrix
    ARROW_DRAW      = 0.4,    # animated arrow creation
    MATRIX_FILL     = 1.5,    # fill full matrix cell-by-cell
    MATRIX_ROW      = 0.08,   # delay between rows during fill
    MATRIX_CELL     = 0.02,   # delay between cells in one row
    DOT_PRODUCT     = 0.6,    # Q·K dot product animation
    SOFTMAX         = 0.5,    # softmax curve animation
    FLOW_THROUGH    = 0.8,    # data token travels through a block
    LAYER_PASS      = 0.4,    # one transformer layer lights up
    HIGHLIGHT       = 0.3,    # highlight / glow pulse
    LABEL_FADE      = 0.25,   # math label appears
    PAUSE_SHORT     = 0.5,    # brief hold between steps
    PAUSE_NORMAL    = 1.0,    # standard pause for viewer to read
    PAUSE_LONG      = 2.0,    # long pause before next section

    # Looping / autoregressive
    GENERATION_STEP = 1.2,    # time per generated token
    KV_CACHE_SHOW   = 0.3,    # one cache slot fills
)


# ---------------------------------------------------------------------------
# LAYOUT  — positions, spacing, alignment
# ---------------------------------------------------------------------------

LAYOUT = SimpleNamespace(
    # Standard spacing units
    XS      = 0.15,
    SM      = 0.25,
    MD      = 0.5,
    LG      = 1.0,
    XL      = 1.5,
    XXL     = 2.5,

    # Common vertical positions (y-axis, centered at 0)
    TOP             =  3.2,
    UPPER           =  2.0,
    CENTER          =  0.0,
    LOWER           = -2.0,
    BOTTOM          = -3.2,

    # Common horizontal positions (x-axis)
    FAR_LEFT        = -5.5,
    LEFT            = -3.5,
    MID_LEFT        = -1.8,
    MID_RIGHT       =  1.8,
    RIGHT           =  3.5,
    FAR_RIGHT       =  5.5,

    # Multi-column helper offsets (3-column layout)
    COL3_LEFT       = -4.2,
    COL3_CENTER     =  0.0,
    COL3_RIGHT      =  4.2,

    # Label offsets from parent object
    LABEL_ABOVE     =  0.35,   # default label y-offset above shape
    LABEL_BELOW     = -0.35,   # default label y-offset below shape
    LABEL_RIGHT     =  0.35,
    LABEL_LEFT      = -0.35,

    # Arrow
    ARROW_BUFF      =  0.12,   # gap between arrow tip and target
    ARROW_TIP_SIZE  =  0.2,

    # Side panel width (for accompanying explanation boxes)
    PANEL_WIDTH     =  3.8,
)


# ---------------------------------------------------------------------------
# TYPOGRAPHY  — text sizing
# ---------------------------------------------------------------------------

TYPOGRAPHY = SimpleNamespace(
    # Scale (Manim font_size units)
    TITLE       = 42,
    HEADING     = 32,
    SUBHEADING  = 24,
    BODY        = 20,
    CAPTION     = 16,
    SMALL       = 13,
    TINY        = 10,   # matrix cell labels, dimension braces

    # Token box text
    TOKEN_WORD  = 22,
    TOKEN_ID    = 12,   # token-id subscript

    # Math / formula text
    FORMULA     = 22,
    FORMULA_SM  = 16,

    # Stroke widths
    STROKE_THIN     = 1.0,
    STROKE_NORMAL   = 2.0,
    STROKE_THICK    = 3.5,
    STROKE_HEAVY    = 5.0,
)


# ---------------------------------------------------------------------------
# MATRIX  — sizes for grid / matrix visuals
# ---------------------------------------------------------------------------

MATRIX = SimpleNamespace(
    # Default cell dimensions (Manim units)
    CELL_SIZE       = 0.38,   # square cell side length
    CELL_SMALL      = 0.22,   # compact matrix cell
    CELL_LARGE      = 0.55,   # large/zoomed cell

    # Cell border
    CELL_STROKE     = 1.0,
    CELL_RADIUS     = 0.04,   # corner rounding on cells

    # Opacity
    CELL_FILL_OPACITY       = 0.85,
    CELL_DIM_OPACITY        = 0.2,
    CELL_HIGHLIGHT_OPACITY  = 1.0,

    # Maximum cells to render (beyond this, fade to "...")
    MAX_ROWS        = 12,
    MAX_COLS        = 12,

    # Label format
    SHOW_VALUES     = True,    # show float values inside cells
    VALUE_DECIMALS  = 2,       # decimal places for displayed floats
)


# ---------------------------------------------------------------------------
# TOKEN  — token box geometry
# ---------------------------------------------------------------------------

TOKEN = SimpleNamespace(
    BOX_WIDTH       = 1.1,    # token box width  (Manim units)
    BOX_HEIGHT      = 0.55,   # token box height
    BOX_RADIUS      = 0.12,   # rounded corner radius
    BOX_STROKE      = 2.0,
    SPACING         = 0.15,   # gap between adjacent token boxes
    ID_OFFSET       = 0.38,   # token-ID label y-offset below box

    GLOW_RADIUS     = 0.08,   # extra glow spread on highlight
    GLOW_OPACITY    = 0.6,
)


# ---------------------------------------------------------------------------
# EMBEDDING  — embedding matrix and vector display
# ---------------------------------------------------------------------------

EMBEDDING = SimpleNamespace(
    # Canonical model dimensions (can be overridden per animation)
    VOCAB_SIZE      = 50257,   # GPT-2 default vocab size (display only)
    D_MODEL_SMALL   = 64,      # for visual demos (fits on screen)
    D_MODEL_MEDIUM  = 128,
    D_MODEL_DEFAULT = 768,     # BERT-base / GPT-2 small

    # How many rows/cols to show before truncating
    DISPLAY_ROWS    = 8,       # visible vocab rows in EmbeddingMatrix
    DISPLAY_COLS    = 16,      # visible dimension columns

    # Vector bar
    VECTOR_CELL_W   = 0.28,    # width of each dim cell in VectorBar
    VECTOR_CELL_H   = 0.45,
    VECTOR_MAX_SHOW = 20,      # truncate long vectors beyond this many cells

    # Lookup animation
    LOOKUP_GLOW_COLOR = "#FFEB3B",  # highlight color when row is found
)


# ---------------------------------------------------------------------------
# ATTENTION  — attention mechanism geometry
# ---------------------------------------------------------------------------

ATTENTION = SimpleNamespace(
    # Canonical head / dimension config for demos
    NUM_HEADS       = 8,
    D_K             = 64,      # key/query dimension per head  (d_model/h)
    D_V             = 64,      # value dimension per head
    SEQ_LEN_SHORT   = 6,       # short sequence for demo
    SEQ_LEN_DEFAULT = 12,

    # Score matrix cell
    SCORE_CELL      = 0.40,    # size of each attention score cell

    # Heatmap
    HEATMAP_OPACITY = 0.9,

    # Causal mask
    MASK_OPACITY    = 0.85,    # opacity of masked-out upper triangle
    MASK_COLOR_HEX  = "#111111",

    # Head layout (multi-head grid)
    HEAD_SPACING    = 0.25,    # gap between head panels
    HEAD_LABEL_SIZE = 14,      # "Head 1" label font size

    # Softmax temperature for visualisation default
    TEMP_DEFAULT    = 1.0,
    TEMP_COLD       = 0.3,
    TEMP_HOT        = 3.0,
)


# ---------------------------------------------------------------------------
# FFN  — feed-forward network
# ---------------------------------------------------------------------------

FFN = SimpleNamespace(
    EXPANSION_FACTOR    = 4,        # d_ff = d_model × EXPANSION_FACTOR
    D_FF_DEFAULT        = 3072,     # 768 × 4 for BERT-base

    # Node / bar sizes for FFNBlock diagram
    NODE_RADIUS         = 0.12,
    NODE_SPACING        = 0.22,
    BAR_WIDTH           = 0.18,     # width of each neuron bar

    MAX_NODES_SHOW      = 16,       # cap visible neurons before "..."

    # MoE
    NUM_EXPERTS         = 8,
    TOP_K_EXPERTS       = 2,
    EXPERT_BOX_W        = 0.9,
    EXPERT_BOX_H        = 0.5,
)


# ---------------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------------

NORM = SimpleNamespace(
    # RMS / LayerNorm visual bar
    BAR_HEIGHT      = 0.35,
    BAR_SPACING     = 0.05,

    # Residual connection
    RESIDUAL_ARC_HEIGHT = 0.6,    # how far the skip arc bows out
    RESIDUAL_COLOR_HEX  = "#8BC34A",
)


# ---------------------------------------------------------------------------
# SAMPLING / OUTPUT
# ---------------------------------------------------------------------------

SAMPLING = SimpleNamespace(
    LOGIT_BAR_WIDTH     = 0.3,    # bar chart bar width
    LOGIT_BAR_MAX_H     = 2.5,    # maximum bar height
    LOGIT_BAR_SPACING   = 0.08,   # gap between bars
    TOP_K_DEFAULT       = 40,
    TOP_P_DEFAULT       = 0.9,
    MAX_BARS_DISPLAY    = 20,     # show only top-N bars to avoid clutter
)


# ---------------------------------------------------------------------------
# MODEL — common reference architectures (for comparison tables / labels)
# ---------------------------------------------------------------------------

MODEL = SimpleNamespace(
    GPT2_SMALL  = dict(d_model=768,  heads=12, layers=12, params="117M"),
    GPT2_MEDIUM = dict(d_model=1024, heads=16, layers=24, params="345M"),
    GPT2_LARGE  = dict(d_model=1280, heads=20, layers=36, params="774M"),
    GPT2_XL     = dict(d_model=1600, heads=25, layers=48, params="1.5B"),

    BERT_BASE   = dict(d_model=768,  heads=12, layers=12, params="110M"),
    BERT_LARGE  = dict(d_model=1024, heads=16, layers=24, params="340M"),

    LLAMA3_8B   = dict(d_model=4096, heads=32, layers=32, params="8B"),
    LLAMA3_70B  = dict(d_model=8192, heads=64, layers=80, params="70B"),

    # Tiny demo model (fits on screen without truncation)
    DEMO        = dict(d_model=64,   heads=4,  layers=2,  params="demo"),
)


# ---------------------------------------------------------------------------
# Z-INDEX  — layering order for overlapping Manim objects
# ---------------------------------------------------------------------------

Z_INDEX = SimpleNamespace(
    BACKGROUND      = 0,
    GRID            = 1,
    MATRIX_CELL     = 2,
    ARROW           = 3,
    SHAPE           = 4,
    HIGHLIGHT       = 5,
    LABEL           = 6,
    FORMULA         = 7,
    TOOLTIP         = 8,
    OVERLAY         = 9,
)


# ---------------------------------------------------------------------------
# EASING  — Manim rate_func aliases (import from manim and alias here)
# ---------------------------------------------------------------------------
# Usage: animate(..., rate_func=EASING.SMOOTH)

try:
    from manim import (
        smooth,
        linear,
        ease_in_sine,
        ease_out_sine,
        ease_in_out_sine,
        ease_in_cubic,
        ease_out_cubic,
        rush_into,
        rush_from,
        there_and_back,
    )
    EASING = SimpleNamespace(
        SMOOTH          = smooth,
        LINEAR          = linear,
        EASE_IN         = ease_in_sine,
        EASE_OUT        = ease_out_sine,
        EASE_IN_OUT     = ease_in_out_sine,
        EASE_IN_CUBIC   = ease_in_cubic,
        EASE_OUT_CUBIC  = ease_out_cubic,
        RUSH_IN         = rush_into,
        RUSH_OUT        = rush_from,
        PULSE           = there_and_back,   # good for highlight / glow
    )
except ImportError:
    # Manim not installed — stubs for IDE / linting
    EASING = SimpleNamespace()


# ---------------------------------------------------------------------------
# QUICK REFERENCE  (python -m llmanim.styles.constants)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import textwrap

    namespaces = {
        "SCENE":      SCENE,
        "TIMING":     TIMING,
        "LAYOUT":     LAYOUT,
        "TYPOGRAPHY": TYPOGRAPHY,
        "MATRIX":     MATRIX,
        "TOKEN":      TOKEN,
        "EMBEDDING":  EMBEDDING,
        "ATTENTION":  ATTENTION,
        "FFN":        FFN,
        "NORM":       NORM,
        "SAMPLING":   SAMPLING,
        "Z_INDEX":    Z_INDEX,
    }

    print("=" * 60)
    print("  llmanim  —  Constants Reference")
    print("=" * 60)

    for ns_name, ns in namespaces.items():
        print(f"\n  {ns_name}")
        print("  " + "-" * (len(ns_name) + 2))
        for k, v in vars(ns).items():
            line = f"    {k:<28} {v}"
            print(textwrap.shorten(line, width=70, placeholder=" ..."))