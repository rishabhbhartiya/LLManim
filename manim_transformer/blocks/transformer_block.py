"""
blocks/transformer_block.py
============================
Manim assets for visualizing a complete Transformer block and stacks of them.

Assets
------
SubLayerBox          — Single labelled sublayer rectangle (MHA, FFN, LN, …)
ResidualStream       — Vertical input→output stream with a skip-connection arc
TransformerBlockDiagram — Full block assembled from sublayers + residuals
                          Compact ↔ detailed toggle; per-sublayer zoom
EncoderBlockAnim     — Sequential data-flow through an Encoder block
DecoderBlockAnim     — Decoder block with masked-MHA + cross-attention
GPTBlockAnim         — GPT-style causal block (pre-norm variant)
LayerStackAnim       — N blocks stacked; data ripples through each layer

Design conventions (matches embedding_lookup.py / positional_encoding.py)
--------------------------------------------------------------------------
  • Every class extends VGroup — composable, positionable, scalable
  • Every public animation method returns a Manim Animation / Succession
  • Colors from styles.colors with graceful fallback
  • _np alias for numpy

Usage
-----
    from blocks.transformer_block import TransformerBlockDiagram, LayerStackAnim

    class BlockScene(Scene):
        def construct(self):
            block = TransformerBlockDiagram(style="gpt")
            self.add(block)
            self.play(block.flow_animation())
            self.wait()
"""

from __future__ import annotations

from manim import (
    VGroup, Rectangle, RoundedRectangle, Text, MathTex,
    Arrow, CurvedArrow, Line, ArcBetweenPoints,
    Scene, Animation, AnimationGroup, Succession,
    FadeIn, FadeOut, Create, Write, Uncreate,
    Transform, ReplacementTransform,
    Indicate, Flash, Circumscribe,
    LaggedStart,
    GrowArrow, MoveToTarget,
    LEFT, RIGHT, UP, DOWN, ORIGIN, UL, UR, DL, DR,
    WHITE, BLACK, GRAY, DARK_GRAY, LIGHT_GRAY,
    RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, TEAL,
    interpolate_color,
    rate_functions, PI,
    Brace, BraceBetweenPoints,
    Dot, DashedLine,
)

import numpy as _np

# ---------------------------------------------------------------------------
# Style imports — graceful fallback
# ---------------------------------------------------------------------------
try:
    from styles.colors import COLORS
except ImportError:
    COLORS = {
        "token":        "#4CAF50",
        "embedding":    "#2196F3",
        "attention":    "#9C27B0",
        "ffn":          "#FF9800",
        "norm":         "#00BCD4",
        "output":       "#F44336",
        "query":        "#E91E63",
        "key":          "#3F51B5",
        "value":        "#009688",
        "weight_matrix":"#795548",
        "highlight":    "#FFEB3B",
        "arrow":        "#FFFFFF",
        "dim":          "#555555",
    }

try:
    from styles.constants import (
        CELL_SIZE, SMALL_CELL, LABEL_SCALE,
        DEFAULT_RUN_TIME, FAST_RUN_TIME,
    )
except ImportError:
    CELL_SIZE        = 0.45
    SMALL_CELL       = 0.28
    LABEL_SCALE      = 0.35
    DEFAULT_RUN_TIME = 1.0
    FAST_RUN_TIME    = 0.4


# ---------------------------------------------------------------------------
# Layout constants (tweak here to rescale everything at once)
# ---------------------------------------------------------------------------
BOX_W         = 2.8      # sublayer box width
BOX_H         = 0.55     # sublayer box height
BOX_GAP       = 0.30     # vertical gap between boxes
STREAM_X      = -1.80    # x-position of the residual stream line
SKIP_CTRL_DX  = 0.90     # horizontal bulge of skip-connection arc
ADD_DOT_R     = 0.14     # radius of the ⊕ addition node
STACK_GAP     = 0.45     # extra gap between stacked blocks


# ---------------------------------------------------------------------------
# Sublayer type → visual style
# ---------------------------------------------------------------------------
_SUBLAYER_STYLES: dict[str, dict] = {
    "MHA":       {"fill": COLORS["attention"], "label": "Multi-Head\nAttention"},
    "MaskedMHA": {"fill": COLORS["attention"], "label": "Masked\nMHA"},
    "CrossAttn": {"fill": COLORS["key"],       "label": "Cross\nAttention"},
    "FFN":       {"fill": COLORS["ffn"],       "label": "Feed\nForward"},
    "LN":        {"fill": COLORS["norm"],      "label": "Layer\nNorm"},
    "LN1":       {"fill": COLORS["norm"],      "label": "Layer\nNorm 1"},
    "LN2":       {"fill": COLORS["norm"],      "label": "Layer\nNorm 2"},
    "Add":       {"fill": COLORS["output"],    "label": "Add &\nNorm"},
}


# ===========================================================================
# 1.  SubLayerBox
# ===========================================================================

class SubLayerBox(VGroup):
    """
    A single labelled rounded rectangle representing one sublayer.

    Parameters
    ----------
    kind      : str   — key into _SUBLAYER_STYLES ('MHA', 'FFN', 'LN', …)
    width     : float
    height    : float
    label     : str | None  — override the default label
    color     : str | None  — override the default fill color
    """

    def __init__(
        self,
        kind:   str,
        width:  float      = BOX_W,
        height: float      = BOX_H,
        label:  str | None = None,
        color:  str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        style      = _SUBLAYER_STYLES.get(kind, {"fill": GRAY, "label": kind})
        fill_color = color or style["fill"]
        text_str   = label or style["label"]

        self._rect = RoundedRectangle(
            width=width, height=height,
            corner_radius=0.08,
            fill_color=fill_color,
            fill_opacity=0.82,
            stroke_color=WHITE,
            stroke_width=1.2,
        )
        # two-line labels need smaller scale
        n_lines = text_str.count("\n") + 1
        scale   = LABEL_SCALE if n_lines < 2 else LABEL_SCALE - 0.04

        self._label = Text(text_str, font="Monospace") \
                          .scale(scale) \
                          .set_color(WHITE) \
                          .move_to(self._rect.get_center())

        self.add(self._rect, self._label)
        self.kind       = kind
        self.fill_color = fill_color

    # ── geometry helpers ─────────────────────────────────────────────────────

    def get_top_center(self):
        return self._rect.get_top()

    def get_bottom_center(self):
        return self._rect.get_bottom()

    # ── animations ───────────────────────────────────────────────────────────

    def appear(self, run_time: float = FAST_RUN_TIME) -> FadeIn:
        return FadeIn(self, shift=UP * 0.1, run_time=run_time)

    def activate(
        self,
        color:    str   = COLORS["highlight"],
        run_time: float = FAST_RUN_TIME * 1.5,
    ) -> AnimationGroup:
        """Flash + brighten to signal data is passing through."""
        return AnimationGroup(
            Flash(self._rect, color=color, line_length=0.18, run_time=run_time),
            self._rect.animate(run_time=run_time).set_stroke(color=color, width=2.8),
            lag_ratio=0.0,
        )

    def deactivate(self, run_time: float = FAST_RUN_TIME) -> Animation:
        """Return stroke to default after activation."""
        return self._rect.animate(run_time=run_time) \
                   .set_stroke(color=WHITE, width=1.2)

    def zoom_label(self, scale: float = 1.35, run_time: float = FAST_RUN_TIME) -> Animation:
        """Temporarily scale up the label — for emphasis."""
        return self._label.animate(run_time=run_time).scale(scale)

    def highlight_border(self, color: str = COLORS["highlight"]) -> Animation:
        return self._rect.animate.set_stroke(color=color, width=3.0)


# ===========================================================================
# 2.  _AddNode  (internal — the ⊕ circle used at residual junctions)
# ===========================================================================

class _AddNode(VGroup):
    def __init__(self, radius: float = ADD_DOT_R, **kwargs):
        super().__init__(**kwargs)
        circle = RoundedRectangle(
            width=radius * 2.2, height=radius * 2.2,
            corner_radius=radius,
            fill_color=COLORS["output"],
            fill_opacity=0.85,
            stroke_color=WHITE,
            stroke_width=1.0,
        )
        plus = MathTex(r"\oplus", font_size=16).set_color(WHITE).move_to(circle)
        self.add(circle, plus)
        self._circle = circle

    def activate(self, run_time: float = FAST_RUN_TIME) -> Flash:
        return Flash(self._circle, color=COLORS["highlight"],
                     line_length=0.12, run_time=run_time)


# ===========================================================================
# 3.  ResidualStream
# ===========================================================================

class ResidualStream(VGroup):
    """
    The vertical backbone of a Transformer block:
      • a straight vertical line (the residual stream)
      • a curved arc that jumps over each sublayer
      • an ⊕ node where the arc reconnects

    Parameters
    ----------
    sublayer    : SubLayerBox — the sublayer this skip connection bypasses
    stream_x    : float       — x-position of the main stream line
    arc_ctrl_dx : float       — how far right the arc bows out
    """

    def __init__(
        self,
        sublayer:    SubLayerBox,
        stream_x:    float = STREAM_X,
        arc_ctrl_dx: float = SKIP_CTRL_DX,
        **kwargs,
    ):
        super().__init__(**kwargs)

        top_y    = sublayer.get_top_center()[1]    + BOX_GAP * 0.5
        bot_y    = sublayer.get_bottom_center()[1] - BOX_GAP * 0.5

        top_pt   = _np.array([stream_x, top_y, 0])
        bot_pt   = _np.array([stream_x, bot_y, 0])

        # straight stream segment
        self._stream_seg = Line(top_pt, bot_pt, color=COLORS["arrow"], stroke_width=1.4)

        # curved skip arc (right of the stream line)
        arc_top = _np.array([stream_x, top_y, 0])
        arc_bot = _np.array([stream_x, bot_y, 0])
        self._arc = ArcBetweenPoints(
            arc_top, arc_bot,
            angle=-PI * 0.55,   # negative → bows to the RIGHT
            color=COLORS["embedding"],
            stroke_width=1.5,
        )

        # ⊕ addition node at the bottom reconnection
        self._add_node = _AddNode()
        self._add_node.move_to(bot_pt)

        # small downward arrow from ⊕ to the next element
        arrow_tip = bot_pt + DOWN * (ADD_DOT_R + 0.18)
        self._out_arrow = Arrow(
            bot_pt, arrow_tip,
            buff=ADD_DOT_R, color=COLORS["arrow"],
            stroke_width=1.2, max_tip_length_to_length_ratio=0.4,
        )

        self.add(self._stream_seg, self._arc, self._add_node, self._out_arrow)

    # ── animations ───────────────────────────────────────────────────────────

    def appear(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        return Succession(
            Create(self._stream_seg, run_time=run_time * 0.4),
            Create(self._arc,        run_time=run_time * 0.4),
            FadeIn(self._add_node,   run_time=run_time * 0.2),
        )

    def animate_skip(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """Trace the skip path: stream down → arc → ⊕ pulse."""
        return Succession(
            Create(self._arc, run_time=run_time * 0.6,
                   rate_func=rate_functions.ease_in_out_sine),
            self._add_node.activate(run_time=run_time * 0.4),
        )

    def animate_through(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """Animate data flowing straight through (not via skip arc)."""
        return Succession(
            Create(self._stream_seg, run_time=run_time * 0.5),
            GrowArrow(self._out_arrow, run_time=run_time * 0.5),
        )


# ===========================================================================
# 4.  TransformerBlockDiagram
# ===========================================================================

class TransformerBlockDiagram(VGroup):
    """
    Full Transformer block assembled from SubLayerBoxes + ResidualStreams.

    Supports three style presets:
        "encoder" — LN → MHA → Add  → LN → FFN → Add
        "decoder" — LN → MaskedMHA → Add → LN → CrossAttn → Add → LN → FFN → Add
        "gpt"     — LN → MHA → Add  → LN → FFN → Add  (pre-norm, causal)

    Parameters
    ----------
    style       : str   — 'encoder' | 'decoder' | 'gpt'
    box_width   : float
    box_height  : float
    show_labels : bool  — show dimension annotations on arrows
    """

    # sublayer sequences per style
    _SEQUENCES = {
        "encoder": ["LN", "MHA",       "Add", "LN", "FFN", "Add"],
        "decoder": ["LN", "MaskedMHA", "Add", "LN", "CrossAttn", "Add", "LN", "FFN", "Add"],
        "gpt":     ["LN", "MHA",       "Add", "LN", "FFN", "Add"],
    }

    def __init__(
        self,
        style:       str   = "gpt",
        box_width:   float = BOX_W,
        box_height:  float = BOX_H,
        show_labels: bool  = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.style = style
        seq = self._SEQUENCES.get(style, self._SEQUENCES["gpt"])

        self._boxes:   list[SubLayerBox]  = []
        self._arrows:  list[Arrow]        = []
        self._skips:   list[ResidualStream] = []

        # ── 1. Build sublayer boxes top → bottom ──────────────────────────────
        y_cursor = 0.0
        for kind in seq:
            box = SubLayerBox(kind, width=box_width, height=box_height)
            box.move_to(_np.array([0.0, -y_cursor, 0]))
            self._boxes.append(box)
            y_cursor += box_height + BOX_GAP

        self.add(*self._boxes)

        # ── 2. Connect boxes with arrows ──────────────────────────────────────
        for i in range(len(self._boxes) - 1):
            top_box = self._boxes[i]
            bot_box = self._boxes[i + 1]
            arr = Arrow(
                top_box.get_bottom_center(),
                bot_box.get_top_center(),
                buff=0.05,
                color=COLORS["arrow"],
                stroke_width=1.3,
                max_tip_length_to_length_ratio=0.35,
            )
            self._arrows.append(arr)
            self.add(arr)

        # ── 3. Add residual skip connections around MHA/FFN pairs ─────────────
        #   A "skip" jumps from just above an LN to just below the paired Add
        self._build_residuals(seq, box_width)

        # ── 4. Input / output labels ──────────────────────────────────────────
        in_arrow = Arrow(
            self._boxes[0].get_top_center() + UP * 0.6,
            self._boxes[0].get_top_center(),
            buff=0.05, color=COLORS["arrow"], stroke_width=1.3,
        )
        out_arrow = Arrow(
            self._boxes[-1].get_bottom_center(),
            self._boxes[-1].get_bottom_center() + DOWN * 0.6,
            buff=0.05, color=COLORS["arrow"], stroke_width=1.3,
        )
        in_lbl  = Text("x  (input)", font="Monospace") \
                      .scale(LABEL_SCALE).set_color(COLORS["token"]) \
                      .next_to(in_arrow, UP, buff=0.08)
        out_lbl = Text("x' (output)", font="Monospace") \
                      .scale(LABEL_SCALE).set_color(COLORS["output"]) \
                      .next_to(out_arrow, DOWN, buff=0.08)

        self._io_group = VGroup(in_arrow, out_arrow, in_lbl, out_lbl)
        self.add(self._io_group)

        # ── 5. Block border brace on the left ─────────────────────────────────
        if show_labels:
            brace = Brace(VGroup(*self._boxes), LEFT, buff=0.18)
            brace_lbl = brace.get_text(
                f"Transformer Block\n({style.upper()})"
            ).scale(LABEL_SCALE).set_color(LIGHT_GRAY)
            self._brace_group = VGroup(brace, brace_lbl)
            self.add(self._brace_group)

    # ── internal: build skip connections ─────────────────────────────────────

    def _build_residuals(self, seq: list[str], box_width: float):
        """
        Identify (LN or start) → Add pairs and draw a skip arc around them.
        Strategy: any 'Add' box gets a skip from the box just before its
        preceding LN sibling.
        """
        stream_x = -(box_width / 2 + 0.55)

        # Find groups: every 'Add' gets a skip from the entry above the LN
        i = 0
        while i < len(seq):
            if seq[i] == "Add":
                # find the LN that precedes this Add
                ln_idx = i - 1
                while ln_idx >= 0 and seq[ln_idx] not in ("LN", "LN1", "LN2"):
                    ln_idx -= 1
                # skip goes from above LN box to below Add box
                skip = ResidualStream(
                    sublayer=_SpanBox(
                        self._boxes[max(ln_idx, 0)],
                        self._boxes[i],
                    ),
                    stream_x=stream_x,
                )
                self._skips.append(skip)
                self.add(skip)
            i += 1

    # ── public animation methods ──────────────────────────────────────────────

    def appear_animation(self, run_time: float = DEFAULT_RUN_TIME * 2) -> Succession:
        """
        Block materialises: boxes fade in top→bottom, then arrows grow,
        then skip connections draw.
        """
        box_anim = LaggedStart(
            *[box.appear(FAST_RUN_TIME) for box in self._boxes],
            lag_ratio=0.12, run_time=run_time * 0.45,
        )
        arr_anim = LaggedStart(
            *[GrowArrow(a) for a in self._arrows],
            lag_ratio=0.1, run_time=run_time * 0.25,
        )
        skip_anim = LaggedStart(
            *[skip.appear(run_time * 0.4) for skip in self._skips],
            lag_ratio=0.3, run_time=run_time * 0.3,
        )
        return Succession(
            box_anim,
            AnimationGroup(arr_anim, FadeIn(self._io_group), lag_ratio=0.0),
            skip_anim,
        )

    def flow_animation(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """
        Activate sublayers one by one (top → bottom) with a highlight pulse,
        simulating a single forward pass.
        """
        steps: list[Animation] = []
        for box in self._boxes:
            steps.append(box.activate(run_time=run_time))
            steps.append(box.deactivate(run_time=FAST_RUN_TIME * 0.5))
        return Succession(*steps)

    def highlight_sublayer(
        self,
        kind:    str,
        color:   str   = COLORS["highlight"],
        run_time: float = DEFAULT_RUN_TIME,
    ) -> AnimationGroup:
        """
        Highlight all sublayers of a given kind (e.g. all 'LN' boxes).
        Dims the rest.
        """
        anims = []
        for box in self._boxes:
            if box.kind == kind:
                anims.append(box.activate(color=color, run_time=run_time))
            else:
                anims.append(box.animate(run_time=run_time).set_opacity(0.25))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def reset_opacity(self, run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        return AnimationGroup(
            *[box.animate(run_time=run_time).set_opacity(1.0) for box in self._boxes],
            lag_ratio=0.0,
        )

    def zoom_into(
        self,
        kind:      str,
        scale:     float = 2.5,
        run_time:  float = DEFAULT_RUN_TIME,
    ) -> tuple[SubLayerBox | None, AnimationGroup]:
        """
        Scale up the first matching sublayer to fill attention.
        Returns (box, animation).  Call zoom_out() to restore.
        """
        target = next((b for b in self._boxes if b.kind == kind), None)
        if target is None:
            return None, AnimationGroup()
        self._pre_zoom_state = (target, target.get_center(), target.width)
        anim = AnimationGroup(
            target.animate(run_time=run_time).scale(scale),
            *[b.animate(run_time=run_time).set_opacity(0.15)
              for b in self._boxes if b is not target],
            lag_ratio=0.0,
        )
        return target, anim

    def zoom_out(self, run_time: float = DEFAULT_RUN_TIME) -> AnimationGroup:
        """Restore after zoom_into."""
        if not hasattr(self, "_pre_zoom_state"):
            return AnimationGroup()
        target, original_center, original_width = self._pre_zoom_state
        current_scale = target.width / original_width
        return AnimationGroup(
            target.animate(run_time=run_time).scale(1 / current_scale),
            *[b.animate(run_time=run_time).set_opacity(1.0) for b in self._boxes],
            lag_ratio=0.0,
        )

    def compact_mode(self, run_time: float = DEFAULT_RUN_TIME) -> Animation:
        """Squish all boxes to half height — for overview slides."""
        return AnimationGroup(
            *[
                box.animate(run_time=run_time).stretch_to_fit_height(BOX_H * 0.5)
                for box in self._boxes
            ],
            lag_ratio=0.0,
        )

    def get_box(self, kind: str) -> SubLayerBox | None:
        """Return the first sublayer box of the given kind."""
        return next((b for b in self._boxes if b.kind == kind), None)


# ===========================================================================
# 5.  EncoderBlockAnim
# ===========================================================================

class EncoderBlockAnim(TransformerBlockDiagram):
    """
    Encoder block: LN → MHA → Add → LN → FFN → Add
    Extends TransformerBlockDiagram with encoder-specific narration helpers.
    """

    def __init__(self, **kwargs):
        super().__init__(style="encoder", **kwargs)

    def narrate_self_attention(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """Highlight the attention sublayer with an explanatory label."""
        mha_box = self.get_box("MHA")
        label = Text("Self-Attention:\nquery/key/value from same sequence",
                     font="Monospace") \
                    .scale(LABEL_SCALE - 0.04) \
                    .set_color(COLORS["attention"]) \
                    .next_to(mha_box, RIGHT, buff=0.35)
        return Succession(
            mha_box.activate(color=COLORS["attention"], run_time=run_time),
            FadeIn(label, shift=LEFT * 0.1, run_time=FAST_RUN_TIME),
            FadeOut(label, run_time=FAST_RUN_TIME),
            mha_box.deactivate(run_time=FAST_RUN_TIME),
        )

    def narrate_ffn(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        ffn_box = self.get_box("FFN")
        label = Text("FFN: d_model → 4×d_model → d_model",
                     font="Monospace") \
                    .scale(LABEL_SCALE - 0.04) \
                    .set_color(COLORS["ffn"]) \
                    .next_to(ffn_box, RIGHT, buff=0.35)
        return Succession(
            ffn_box.activate(color=COLORS["ffn"], run_time=run_time),
            FadeIn(label, shift=LEFT * 0.1, run_time=FAST_RUN_TIME),
            FadeOut(label, run_time=FAST_RUN_TIME),
            ffn_box.deactivate(run_time=FAST_RUN_TIME),
        )


# ===========================================================================
# 6.  DecoderBlockAnim
# ===========================================================================

class DecoderBlockAnim(TransformerBlockDiagram):
    """
    Decoder block:
        LN → MaskedMHA → Add → LN → CrossAttn → Add → LN → FFN → Add

    The cross-attention sublayer receives encoder output (K, V) from the side.
    """

    def __init__(self, **kwargs):
        super().__init__(style="decoder", **kwargs)
        self._encoder_arrow: Arrow | None = None

    def add_encoder_input(self, run_time: float = DEFAULT_RUN_TIME) -> tuple[VGroup, Succession]:
        """
        Draw an arrow entering the cross-attention box from the right,
        labelled 'Encoder K, V'.
        Returns (arrow_group, Succession animation).
        """
        cross_box = self.get_box("CrossAttn")
        if cross_box is None:
            return VGroup(), Succession()

        start = cross_box.get_right() + RIGHT * 1.8
        end   = cross_box.get_right()

        arrow = Arrow(
            start, end, buff=0.05,
            color=COLORS["key"], stroke_width=1.5,
            max_tip_length_to_length_ratio=0.3,
        )
        label = Text("Encoder\nK, V", font="Monospace") \
                    .scale(LABEL_SCALE) \
                    .set_color(COLORS["key"]) \
                    .next_to(arrow, UP, buff=0.08)
        group = VGroup(arrow, label)
        self._encoder_arrow = group
        self.add(group)
        return group, Succession(GrowArrow(arrow), FadeIn(label))

    def narrate_masked_attention(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        box   = self.get_box("MaskedMHA")
        label = Text("Causal mask: future tokens blocked",
                     font="Monospace") \
                    .scale(LABEL_SCALE - 0.04) \
                    .set_color(COLORS["attention"]) \
                    .next_to(box, RIGHT, buff=0.35)
        return Succession(
            box.activate(color=COLORS["attention"], run_time=run_time),
            FadeIn(label, run_time=FAST_RUN_TIME),
            FadeOut(label, run_time=FAST_RUN_TIME),
            box.deactivate(run_time=FAST_RUN_TIME),
        )

    def narrate_cross_attention(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        box   = self.get_box("CrossAttn")
        label = Text("Query from decoder\nK, V from encoder",
                     font="Monospace") \
                    .scale(LABEL_SCALE - 0.04) \
                    .set_color(COLORS["key"]) \
                    .next_to(box, RIGHT, buff=0.35)
        return Succession(
            box.activate(color=COLORS["key"], run_time=run_time),
            FadeIn(label, run_time=FAST_RUN_TIME),
            FadeOut(label, run_time=FAST_RUN_TIME),
            box.deactivate(run_time=FAST_RUN_TIME),
        )


# ===========================================================================
# 7.  GPTBlockAnim
# ===========================================================================

class GPTBlockAnim(TransformerBlockDiagram):
    """
    GPT-style causal decoder block (no cross-attention):
        LN → MHA (causal) → Add → LN → FFN → Add
    Pre-norm variant (LN before each sublayer).
    """

    def __init__(self, **kwargs):
        super().__init__(style="gpt", **kwargs)

    def annotate_prenorm(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """Draw a brace + label showing the pre-norm placement."""
        ln_boxes = [b for b in self._boxes if b.kind == "LN"]
        steps: list[Animation] = []
        for ln_box in ln_boxes:
            label = Text("Pre-Norm", font="Monospace") \
                        .scale(LABEL_SCALE - 0.04) \
                        .set_color(COLORS["norm"]) \
                        .next_to(ln_box, LEFT, buff=0.30)
            steps.append(
                AnimationGroup(
                    ln_box.activate(color=COLORS["norm"], run_time=run_time),
                    FadeIn(label, run_time=FAST_RUN_TIME),
                    lag_ratio=0.3,
                )
            )
            steps.append(FadeOut(label, run_time=FAST_RUN_TIME))
        return Succession(*steps)

    def annotate_causal(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        mha = self.get_box("MHA")
        label = Text("Causal (masked)\nself-attention",
                     font="Monospace") \
                    .scale(LABEL_SCALE - 0.04) \
                    .set_color(COLORS["attention"]) \
                    .next_to(mha, RIGHT, buff=0.32)
        return Succession(
            mha.activate(color=COLORS["attention"], run_time=run_time),
            FadeIn(label, run_time=FAST_RUN_TIME),
            FadeOut(label, run_time=FAST_RUN_TIME),
            mha.deactivate(run_time=FAST_RUN_TIME),
        )


# ===========================================================================
# 8.  LayerStackAnim
# ===========================================================================

class LayerStackAnim(VGroup):
    """
    N identical Transformer blocks stacked vertically.
    Data ripples bottom-to-top (as in a typical diagram orientation)
    or top-to-bottom depending on `direction`.

    Parameters
    ----------
    n_layers    : int   — number of blocks in the stack
    style       : str   — 'encoder' | 'decoder' | 'gpt'
    compact     : bool  — use compact (half-height) boxes for each block
    direction   : str   — 'up' (embeddings → top) | 'down' (default for GPT diagrams)
    show_counter: bool  — show "Layer k / N" label that advances during flow
    """

    def __init__(
        self,
        n_layers:     int   = 4,
        style:        str   = "gpt",
        compact:      bool  = True,
        direction:    str   = "down",
        show_counter: bool  = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_layers    = n_layers
        self.style       = style
        self._blocks:    list[TransformerBlockDiagram] = []
        self._separators: list[Line] = []

        sign = 1 if direction == "down" else -1

        # ── build each block ──────────────────────────────────────────────────
        for k in range(n_layers):
            if compact:
                block = _CompactBlock(style=style, layer_num=k + 1, n_layers=n_layers)
            else:
                block = TransformerBlockDiagram(style=style, show_labels=False)

            y_pos = -sign * k * (block.height + STACK_GAP)
            block.move_to(_np.array([0.0, y_pos, 0]))
            self._blocks.append(block)
            self.add(block)

        # ── separators between blocks ─────────────────────────────────────────
        for k in range(n_layers - 1):
            b_top = self._blocks[k]
            b_bot = self._blocks[k + 1]
            mid_y = (b_top.get_bottom()[1] + b_bot.get_top()[1]) / 2
            sep   = DashedLine(
                LEFT * (BOX_W / 2 + 0.2),
                RIGHT * (BOX_W / 2 + 0.2),
                color=COLORS["dim"],
                stroke_width=0.8,
                dash_length=0.08,
            ).move_to(_np.array([0, mid_y, 0]))
            self._separators.append(sep)
            self.add(sep)

        # ── inter-block arrows ────────────────────────────────────────────────
        self._inter_arrows: list[Arrow] = []
        for k in range(n_layers - 1):
            b_top = self._blocks[k]
            b_bot = self._blocks[k + 1]
            arr   = Arrow(
                b_top.get_bottom(),
                b_bot.get_top(),
                buff=0.05,
                color=COLORS["arrow"],
                stroke_width=1.2,
                max_tip_length_to_length_ratio=0.3,
            )
            self._inter_arrows.append(arr)
            self.add(arr)

        # ── input / output labels ─────────────────────────────────────────────
        first, last = self._blocks[0], self._blocks[-1]
        self._in_arr = Arrow(
            first.get_top() + UP * 0.55, first.get_top(),
            buff=0.05, color=COLORS["token"], stroke_width=1.2,
        )
        self._out_arr = Arrow(
            last.get_bottom(), last.get_bottom() + DOWN * 0.55,
            buff=0.05, color=COLORS["output"], stroke_width=1.2,
        )
        in_lbl  = Text("Embeddings + PE", font="Monospace") \
                      .scale(LABEL_SCALE).set_color(COLORS["token"]) \
                      .next_to(self._in_arr, UP, buff=0.06)
        out_lbl = Text("Final hidden states", font="Monospace") \
                      .scale(LABEL_SCALE).set_color(COLORS["output"]) \
                      .next_to(self._out_arr, DOWN, buff=0.06)
        self.add(self._in_arr, self._out_arr, in_lbl, out_lbl)

        # ── layer counter ─────────────────────────────────────────────────────
        if show_counter:
            self._counter_text = Text("", font="Monospace") \
                                     .scale(LABEL_SCALE) \
                                     .set_color(COLORS["highlight"])
            self._counter_text.next_to(self, RIGHT, buff=0.5)
            self.add(self._counter_text)
            self._show_counter = True
        else:
            self._counter_text = None
            self._show_counter = False

    # ── animations ───────────────────────────────────────────────────────────

    def appear_animation(self, run_time: float = DEFAULT_RUN_TIME * 2) -> LaggedStart:
        """All blocks fade in top-to-bottom."""
        block_anims = [FadeIn(b, shift=DOWN * 0.15) for b in self._blocks]
        arrow_anims = [GrowArrow(a) for a in self._inter_arrows]
        return LaggedStart(
            *block_anims,
            *arrow_anims,
            FadeIn(self._in_arr),
            FadeIn(self._out_arr),
            lag_ratio=0.10,
            run_time=run_time,
        )

    def flow_through_animation(
        self,
        run_time_per_layer: float = DEFAULT_RUN_TIME * 0.8,
    ) -> Succession:
        """
        Data pulse travels through every layer in sequence.
        Each block activates fully, then deactivates before the next.
        A counter label updates as each layer lights up.
        """
        steps: list[Animation] = [GrowArrow(self._in_arr)]

        for k, block in enumerate(self._blocks):
            layer_steps: list[Animation] = []

            # update counter
            if self._show_counter and self._counter_text is not None:
                new_counter = Text(
                    f"Layer {k+1} / {self.n_layers}",
                    font="Monospace",
                ).scale(LABEL_SCALE).set_color(COLORS["highlight"]) \
                 .move_to(self._counter_text.get_center())
                layer_steps.append(
                    ReplacementTransform(self._counter_text, new_counter,
                                         run_time=FAST_RUN_TIME)
                )
                self._counter_text = new_counter

            # activate block
            layer_steps.append(block.flow_animation(run_time=run_time_per_layer))

            # inter-block arrow (except last)
            if k < len(self._inter_arrows):
                layer_steps.append(
                    GrowArrow(self._inter_arrows[k], run_time=FAST_RUN_TIME)
                )

            steps.append(Succession(*layer_steps))

        steps.append(GrowArrow(self._out_arr))
        return Succession(*steps)

    def highlight_layer(
        self,
        layer_idx: int,
        color:     str   = COLORS["highlight"],
        run_time:  float = DEFAULT_RUN_TIME,
    ) -> AnimationGroup:
        """Spotlight one layer; dim all others."""
        anims = []
        for k, block in enumerate(self._blocks):
            if k == layer_idx:
                anims.append(Circumscribe(block, color=color, run_time=run_time))
                anims.append(block.animate.set_opacity(1.0))
            else:
                anims.append(block.animate(run_time=run_time).set_opacity(0.2))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def reset_opacity(self, run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        return AnimationGroup(
            *[b.animate(run_time=run_time).set_opacity(1.0) for b in self._blocks],
            lag_ratio=0.0,
        )

    def label_early_late_layers(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """
        Fade in annotations: 'syntax / low-level' on early layers,
        'semantics / reasoning' on later layers.
        """
        early = self._blocks[0]
        late  = self._blocks[-1]
        lbl_e = Text("syntax,\nlow-level", font="Monospace") \
                    .scale(LABEL_SCALE - 0.04) \
                    .set_color(COLORS["token"]) \
                    .next_to(early, LEFT, buff=0.35)
        lbl_l = Text("semantics,\nreasoning", font="Monospace") \
                    .scale(LABEL_SCALE - 0.04) \
                    .set_color(COLORS["attention"]) \
                    .next_to(late, LEFT, buff=0.35)
        return Succession(
            FadeIn(lbl_e, run_time=run_time),
            FadeIn(lbl_l, run_time=run_time),
        )


# ===========================================================================
# Internal helpers — not exported
# ===========================================================================

class _SpanBox:
    """
    Lightweight duck-type that mimics SubLayerBox's geometry interface,
    spanning from the top of `box_top` to the bottom of `box_bot`.
    Used by TransformerBlockDiagram._build_residuals.
    """
    def __init__(self, box_top: SubLayerBox, box_bot: SubLayerBox):
        self._top = box_top
        self._bot = box_bot

    def get_top_center(self):
        return self._top.get_top_center()

    def get_bottom_center(self):
        return self._bot.get_bottom_center()


class _CompactBlock(VGroup):
    """
    A single compressed block for use inside LayerStackAnim.
    Shows the style name + layer number in a slim rounded rect,
    with a left-side colored stripe for the block type.
    """

    _STYLE_COLORS = {
        "encoder": COLORS["embedding"],
        "decoder": COLORS["key"],
        "gpt":     COLORS["attention"],
    }

    def __init__(self, style: str = "gpt", layer_num: int = 1, n_layers: int = 1, **kwargs):
        super().__init__(**kwargs)

        color = self._STYLE_COLORS.get(style, GRAY)

        body = RoundedRectangle(
            width=BOX_W, height=BOX_H * 0.85,
            corner_radius=0.07,
            fill_color=DARK_GRAY,
            fill_opacity=0.88,
            stroke_color=color,
            stroke_width=1.5,
        )
        # left stripe
        stripe = RoundedRectangle(
            width=0.18, height=BOX_H * 0.85,
            corner_radius=0.04,
            fill_color=color,
            fill_opacity=0.95,
            stroke_width=0,
        ).move_to(body.get_left() + RIGHT * 0.09)

        # label: "Layer k  [style]"
        lbl_str = f"Layer {layer_num}   [{style.upper()}]"
        lbl = Text(lbl_str, font="Monospace") \
                  .scale(LABEL_SCALE - 0.04) \
                  .set_color(LIGHT_GRAY) \
                  .move_to(body.get_center())

        self.add(body, stripe, lbl)
        self._body  = body
        self._color = color

    def flow_animation(self, run_time: float = DEFAULT_RUN_TIME * 0.6) -> AnimationGroup:
        """Quick flash on the body border."""
        return AnimationGroup(
            Flash(self._body, color=self._color, line_length=0.18, run_time=run_time),
            self._body.animate(run_time=run_time).set_stroke(color=COLORS["highlight"], width=2.5),
            lag_ratio=0.0,
        )

    def deactivate(self, run_time: float = FAST_RUN_TIME) -> Animation:
        return self._body.animate(run_time=run_time).set_stroke(color=self._color, width=1.5)