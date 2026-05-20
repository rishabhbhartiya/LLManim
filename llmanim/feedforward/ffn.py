"""
feedforward/ffn.py
===================
Manim assets for visualizing the Feed-Forward Network (FFN) sublayer
inside a Transformer block.

Assets
------
FFNBlock             — Full two-layer FFN: expand → activate → compress
ActivationCurve      — Plotted ReLU / GELU / SiLU curves with formula
NeuronFireAnim       — Single neuron: input → gate → output, fires or silent
DimensionExpansionAnim — Narrow vector morphing wide and back, with labels
MoERouterAnim        — Mixture-of-Experts router selecting top-k experts

Design conventions (matches rest of library)
--------------------------------------------
  • Every class extends VGroup — composable, positionable, scalable
  • Every animation method returns Manim Animation / Succession
  • Colors from styles.colors with graceful fallback
  • _np alias for numpy

Usage
-----
    from feedforward.ffn import FFNBlock, ActivationCurve

    class FFNScene(Scene):
        def construct(self):
            ffn = FFNBlock(d_model=512, d_ff=2048)
            self.add(ffn)
            self.play(ffn.build_animation())
            self.wait()
"""

from __future__ import annotations

from manim import (
    VGroup, Rectangle, RoundedRectangle, Polygon, Text, MathTex,
    Arrow, Line, Dot, Axes,
    Scene, Animation, AnimationGroup, Succession,
    FadeIn, FadeOut, Create, Write,
    ReplacementTransform,
    Indicate, Flash, Circumscribe,
    LaggedStart,
    GrowArrow,
    LEFT, RIGHT, UP, DOWN, ORIGIN,
    WHITE, GRAY, DARK_GRAY, LIGHT_GRAY,
    interpolate_color,
    rate_functions, PI,
    Brace,
    DashedLine, DashedVMobject,
)

import numpy as _np

# ---------------------------------------------------------------------------
# Style imports — graceful fallback so file works standalone
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
        SMALL_CELL, LABEL_SCALE,
        DEFAULT_RUN_TIME, FAST_RUN_TIME,
    )
except ImportError:
    SMALL_CELL       = 0.28
    LABEL_SCALE      = 0.35
    DEFAULT_RUN_TIME = 1.0
    FAST_RUN_TIME    = 0.4


# ---------------------------------------------------------------------------
# Activation functions (pure Python — no torch/jax required)
# ---------------------------------------------------------------------------

def _relu(x: float) -> float:
    return max(0.0, float(x))


def _gelu(x: float) -> float:
    x = float(x)
    return 0.5 * x * (1.0 + _np.tanh(_np.sqrt(2.0 / PI) * (x + 0.044715 * x ** 3)))


def _silu(x: float) -> float:
    x = float(x)
    return x / (1.0 + _np.exp(-x))


# name → (callable, LaTeX formula string)
_ACTIVATIONS: dict[str, tuple] = {
    "relu": (_relu, r"\text{ReLU}(x) = \max(0,\,x)"),
    "gelu": (_gelu, r"\text{GELU}(x) \approx x\,\Phi(x)"),
    "silu": (_silu, r"\text{SiLU}(x) = \dfrac{x}{1+e^{-x}}"),
}


# ---------------------------------------------------------------------------
# Shared internal helpers
# ---------------------------------------------------------------------------

def _txt(s: str, color=LIGHT_GRAY, scale: float = LABEL_SCALE) -> Text:
    return Text(s, font="Monospace").scale(scale).set_color(color)


def _dim_brace(mob, direction, label: str, color=LIGHT_GRAY) -> VGroup:
    b   = Brace(mob, direction, buff=0.10, color=color)
    lbl = b.get_text(label).scale(LABEL_SCALE).set_color(color)
    return VGroup(b, lbl)


def _rounded_box(
    label: str,
    color: str,
    w: float = 1.2,
    h: float = 0.46,
    fill_opacity: float = 0.82,
) -> VGroup:
    rect = RoundedRectangle(
        width=w, height=h,
        corner_radius=0.07,
        fill_color=color, fill_opacity=fill_opacity,
        stroke_color=WHITE, stroke_width=1.0,
    )
    n_lines = label.count("\n") + 1
    scale   = LABEL_SCALE if n_lines < 2 else LABEL_SCALE - 0.04
    lbl     = Text(label, font="Monospace").scale(scale).set_color(WHITE)
    lbl.move_to(rect.get_center())
    return VGroup(rect, lbl)


# ===========================================================================
# Internal: _DimBar
# Horizontal cell-bar representing a 1-D vector of some dimension.
# Used by FFNBlock and DimensionExpansionAnim.
# ===========================================================================

class _DimBar(VGroup):
    """
    Horizontal bar of `min(dim, max_cells)` coloured cells.
    Cell colour intensity fades from bright (centre) → dim (edges) to suggest
    a realistic activation distribution without real values.
    """

    def __init__(
        self,
        dim:       int,
        width:     float,
        height:    float,
        color:     str,
        label:     str   = "",
        max_cells: int   = 28,
    ):
        super().__init__()
        self.cells: list[Rectangle] = []
        n  = min(dim, max_cells)
        cs = width / max(n, 1)

        for i in range(n):
            # intensity: bright in the middle, dimmer at the edges
            t    = abs(i / max(n - 1, 1) - 0.5) * 2      # 0 = centre, 1 = edge
            fill = interpolate_color(color, DARK_GRAY, t * 0.62)
            cell = Rectangle(
                width=cs * 0.90, height=height,
                fill_color=fill, fill_opacity=0.85,
                stroke_color=DARK_GRAY, stroke_width=0.25,
            )
            cell.move_to(_np.array([i * cs + cs / 2, 0, 0]))
            self.add(cell)
            self.cells.append(cell)

        if dim > max_cells:
            ell = Text("…", font="Monospace").scale(0.38).set_color(GRAY)
            ell.next_to(self.cells[-1], RIGHT, buff=0.06)
            self.add(ell)

        if label:
            lbl = _txt(label, color, scale=LABEL_SCALE - 0.04)
            lbl.next_to(self, UP, buff=0.10)
            self.add(lbl)

    def appear_animation(self, run_time: float = DEFAULT_RUN_TIME) -> LaggedStart:
        return LaggedStart(
            *[FadeIn(c, scale=0.75) for c in self.cells],
            lag_ratio=0.03,
            run_time=run_time,
        )

    def pulse(self, color: str = COLORS["highlight"], run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        """Flash a sparse sample of cells — signals data passing through."""
        sample = self.cells[:: max(1, len(self.cells) // 8)]
        return AnimationGroup(
            *[Flash(c, color=color, line_length=0.08) for c in sample],
            lag_ratio=0.0,
            run_time=run_time,
        )


# ===========================================================================
# Internal: _TrapezoidArrow
# Filled polygon that widens (expand) or narrows (compress) the data stream.
# ===========================================================================

class _TrapezoidArrow(VGroup):
    """
    A trapezoid whose left edge is `in_h` tall and right edge is `out_h` tall.
    Set `flipped=True` to get a compressing shape (wide → narrow).
    """

    def __init__(
        self,
        in_h:    float,
        out_h:   float,
        width:   float,
        color:   str,
        label:   str  = "",
        flipped: bool = False,
    ):
        super().__init__()

        hi = in_h  * 0.50   # half-height of the input edge
        ho = out_h * 0.88   # half-height of the output edge (larger → expands)

        if not flipped:
            pts = [
                _np.array([-width / 2, -hi, 0]),
                _np.array([ width / 2, -ho, 0]),
                _np.array([ width / 2,  ho, 0]),
                _np.array([-width / 2,  hi, 0]),
            ]
        else:
            pts = [
                _np.array([-width / 2, -ho, 0]),
                _np.array([ width / 2, -hi, 0]),
                _np.array([ width / 2,  hi, 0]),
                _np.array([-width / 2,  ho, 0]),
            ]

        trap = Polygon(*pts,
                       fill_color=color, fill_opacity=0.68,
                       stroke_color=WHITE, stroke_width=0.8)
        self._trap = trap
        self.add(trap)

        if label:
            lbl = _txt(label, color, scale=LABEL_SCALE - 0.05)
            lbl.move_to(trap.get_center())
            self.add(lbl)

    def flow_pulse(self, run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        return AnimationGroup(
            Flash(self._trap, color=COLORS["highlight"],
                  line_length=0.12, run_time=run_time),
            self._trap.animate(run_time=run_time).set_fill(opacity=1.0),
            lag_ratio=0.0,
        )

    def appear_animation(self, run_time: float = FAST_RUN_TIME) -> FadeIn:
        return FadeIn(self, run_time=run_time)


# ===========================================================================
# Internal: _ActivationBand
# A compact rectangle containing a miniature plotted activation curve.
# ===========================================================================

class _ActivationBand(VGroup):
    """Slim panel showing the σ symbol and a tiny plotted curve."""

    def __init__(
        self,
        width:      float,
        height:     float,
        activation: str,
        color:      str,
    ):
        super().__init__()
        self._activation = activation

        body = RoundedRectangle(
            width=width, height=height,
            corner_radius=0.06,
            fill_color=DARK_GRAY, fill_opacity=0.90,
            stroke_color=color, stroke_width=1.6,
        )
        self._body = body
        self.add(body)

        fn, _ = _ACTIVATIONS.get(activation, _ACTIVATIONS["gelu"])

        # miniature axes inside the box
        mini_ax = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-0.3, 1.8, 1],
            x_length=width  * 0.80,
            y_length=height * 0.72,
            axis_config={
                "color": COLORS["dim"], "stroke_width": 0.6,
                "include_tip": False, "include_numbers": False,
            },
            tips=False,
        ).move_to(body.get_center())

        curve = mini_ax.plot(
            fn, x_range=[-2.5, 2.5],
            color=color, stroke_width=1.8,
            use_smoothing=True,
        )
        self._curve = curve
        self.add(mini_ax, curve)

    def appear_animation(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        return Succession(
            FadeIn(self._body, run_time=FAST_RUN_TIME),
            Create(self._curve, run_time=run_time * 0.7),
        )

    def fire_animation(self, run_time: float = DEFAULT_RUN_TIME) -> AnimationGroup:
        return AnimationGroup(
            Flash(self._body, color=COLORS["query"],
                  line_length=0.16, run_time=run_time),
            self._body.animate(run_time=run_time)
                      .set_stroke(color=COLORS["highlight"], width=2.8),
            lag_ratio=0.0,
        )


# ===========================================================================
# 1.  FFNBlock
# ===========================================================================

class FFNBlock(VGroup):
    """
    Full two-layer FFN rendered as a horizontal pipeline:

        [input d_model] ──W₁──> [hidden d_ff] ──σ──> [hidden d_ff] ──W₂──> [output d_model]

    Each stage is a _DimBar; the transitions are _TrapezoidArrows and one
    _ActivationBand.  Dimension braces sit beneath the bars; the formula
    strip floats above.

    Parameters
    ----------
    d_model    : int   — input / output dimension
    d_ff       : int   — hidden (expanded) dimension  (typically 4 × d_model)
    activation : str   — 'relu' | 'gelu' | 'silu'
    bar_height : float — height of each dimension bar
    show_values: bool  — (reserved) show numeric cell values
    """

    def __init__(
        self,
        d_model:    int   = 512,
        d_ff:       int   = 2048,
        activation: str   = "gelu",
        bar_height: float = 0.55,
        show_values: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_model    = d_model
        self.d_ff       = d_ff
        self.activation = activation

        # ── stage widths (log-compressed so d_ff is not comically wide) ─────
        w_input  = 0.65
        w_hidden = 3.20
        w_act    = 0.72
        w_trap   = 0.85

        # ── build each stage ─────────────────────────────────────────────────
        self._input_bar   = _DimBar(d_model, w_input,  bar_height,
                                    COLORS["embedding"], f"d_model\n({d_model})")
        self._expand      = _TrapezoidArrow(bar_height, bar_height,
                                            w_trap, COLORS["ffn"],
                                            label="W₁  b₁")
        self._pre_act_bar = _DimBar(d_ff, w_hidden, bar_height,
                                    COLORS["ffn"], f"d_ff  ({d_ff})")
        self._act_band    = _ActivationBand(w_act, bar_height,
                                            activation, COLORS["query"])
        self._post_act_bar = _DimBar(d_ff, w_hidden, bar_height,
                                     COLORS["query"], "after  σ")
        self._compress    = _TrapezoidArrow(bar_height, bar_height,
                                            w_trap, COLORS["value"],
                                            label="W₂  b₂", flipped=True)
        self._output_bar  = _DimBar(d_model, w_input, bar_height,
                                    COLORS["output"], f"d_model\n({d_model})")

        self._stages = [
            self._input_bar,
            self._expand,
            self._pre_act_bar,
            self._act_band,
            self._post_act_bar,
            self._compress,
            self._output_bar,
        ]

        # ── horizontal layout — each piece placed by running x cursor ────────
        GAP      = 0.10
        x_cursor = 0.0
        for stage in self._stages:
            stage.move_to(_np.array([x_cursor + stage.width / 2, 0, 0]))
            x_cursor += stage.width + GAP

        total_w = x_cursor - GAP
        for stage in self._stages:
            stage.shift(LEFT * total_w / 2)   # centre the whole pipeline

        self.add(*self._stages)

        # ── dimension braces below ────────────────────────────────────────────
        br_in  = _dim_brace(self._input_bar,    DOWN, f"{d_model}",      COLORS["embedding"])
        br_hid = _dim_brace(self._pre_act_bar,  DOWN, f"{d_ff}",         COLORS["ffn"])
        br_out = _dim_brace(self._output_bar,   DOWN, f"{d_model}",      COLORS["output"])
        self._brace_group = VGroup(br_in, br_hid, br_out)
        self.add(self._brace_group)

        # ── formula strip above ───────────────────────────────────────────────
        formula = MathTex(
            r"\text{FFN}(x)=",
            r"\sigma\!\left(",
            r"xW_1+b_1",
            r"\right)",
            r"W_2+b_2",
            font_size=22,
        )
        formula.set_color(LIGHT_GRAY)
        formula[1].set_color(COLORS["query"])   # σ
        formula[2].set_color(COLORS["ffn"])     # xW1+b1
        formula[4].set_color(COLORS["value"])   # W2+b2
        formula.next_to(self, UP, buff=0.35)
        self.add(formula)
        self._formula = formula

        # activation name tag above the band
        act_tag = _txt(f"σ = {activation.upper()}", COLORS["query"])
        act_tag.next_to(self._act_band, UP, buff=0.12)
        self.add(act_tag)

    # ── animations ───────────────────────────────────────────────────────────

    def build_animation(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """
        Left-to-right reveal of each pipeline stage.
        Formula appears first, then stages materialise in order.
        """
        steps: list[Animation] = [Write(self._formula, run_time=run_time * 0.5)]

        for stage in self._stages:
            if hasattr(stage, "appear_animation"):
                steps.append(stage.appear_animation(run_time=run_time * 0.65))
            else:
                steps.append(FadeIn(stage, run_time=FAST_RUN_TIME))

        steps.append(FadeIn(self._brace_group, run_time=FAST_RUN_TIME))
        return Succession(*steps)

    def animate_forward_pass(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """
        Simulate one forward pass — a data pulse travels left → right
        through each stage in sequence.
        """
        return Succession(
            self._input_bar.pulse(run_time=FAST_RUN_TIME),
            self._expand.flow_pulse(run_time=run_time * 0.45),
            self._pre_act_bar.pulse(run_time=FAST_RUN_TIME),
            self._act_band.fire_animation(run_time=run_time * 0.55),
            self._post_act_bar.pulse(run_time=FAST_RUN_TIME),
            self._compress.flow_pulse(run_time=run_time * 0.45),
            self._output_bar.pulse(run_time=FAST_RUN_TIME),
        )

    def highlight_expansion(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """
        Draw a brace + label over the expand step and flash the hidden bar.
        """
        brace = _dim_brace(
            VGroup(self._expand, self._pre_act_bar),
            UP, "4× expansion stores knowledge", COLORS["ffn"],
        )
        return Succession(
            FadeIn(brace, run_time=FAST_RUN_TIME),
            self._pre_act_bar.pulse(COLORS["ffn"], run_time),
            FadeOut(brace, run_time=FAST_RUN_TIME),
        )

    def highlight_compression(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """Draw a brace + label over the compress step."""
        brace = _dim_brace(
            VGroup(self._compress, self._output_bar),
            UP, "compress back to d_model", COLORS["value"],
        )
        return Succession(
            FadeIn(brace, run_time=FAST_RUN_TIME),
            self._compress.flow_pulse(run_time),
            self._output_bar.pulse(COLORS["value"], run_time),
            FadeOut(brace, run_time=FAST_RUN_TIME),
        )

    def swap_activation(
        self,
        new_activation: str,
        run_time: float = DEFAULT_RUN_TIME,
    ) -> ReplacementTransform:
        """
        Morph the activation band to a different function.
        Useful for comparing ReLU vs GELU in a single scene.
        """
        new_band = _ActivationBand(
            width=self._act_band.width,
            height=self._act_band.height,
            activation=new_activation,
            color=COLORS["query"],
        ).move_to(self._act_band.get_center())

        anim = ReplacementTransform(self._act_band, new_band, run_time=run_time)
        self._act_band = new_band
        return anim


# ===========================================================================
# 2.  ActivationCurve
# ===========================================================================

class ActivationCurve(VGroup):
    """
    Plots one or more activation functions on shared axes.

    Parameters
    ----------
    functions   : list[str]          — subset of ['relu', 'gelu', 'silu']
    x_range     : tuple[float,float] — (x_min, x_max)
    axes_w      : float
    axes_h      : float
    show_formula: bool               — display LaTeX formula per curve
    """

    def __init__(
        self,
        functions:    list[str]          = None,
        x_range:      tuple[float,float] = (-3.5, 3.5),
        axes_w:       float = 4.2,
        axes_h:       float = 2.8,
        show_formula: bool  = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if functions is None:
            functions = ["relu", "gelu", "silu"]

        curve_colors = [COLORS["ffn"], COLORS["query"], COLORS["value"]]

        # ── axes ─────────────────────────────────────────────────────────────
        self._axes = Axes(
            x_range=[x_range[0], x_range[1], 1],
            y_range=[-0.8, 2.8, 1],
            x_length=axes_w,
            y_length=axes_h,
            axis_config={
                "color":            COLORS["dim"],
                "stroke_width":     1.2,
                "include_tip":      False,
                "include_numbers":  True,
                "font_size":        14,
            },
            tips=False,
        )
        self.add(self._axes)

        # zero-crossing reference lines
        h_zero = DashedLine(
            self._axes.c2p(x_range[0], 0),
            self._axes.c2p(x_range[1], 0),
            color=COLORS["dim"], stroke_width=0.7, dash_length=0.07,
        )
        v_zero = DashedLine(
            self._axes.c2p(0, -0.8),
            self._axes.c2p(0,  2.8),
            color=COLORS["dim"], stroke_width=0.7, dash_length=0.07,
        )
        self.add(h_zero, v_zero)

        # ── one curve + legend row per function ───────────────────────────────
        self._curves:      list = []
        self._legend_rows: list[VGroup] = []

        for idx, fn_name in enumerate(functions):
            fn, formula_str = _ACTIVATIONS.get(fn_name, _ACTIVATIONS["relu"])
            color = curve_colors[idx % len(curve_colors)]

            curve = self._axes.plot(
                fn,
                x_range=list(x_range),
                color=color,
                stroke_width=2.6,
                use_smoothing=True,
            )
            self._curves.append(curve)
            self.add(curve)

            # legend: colour swatch → name → formula
            swatch = Line(ORIGIN, RIGHT * 0.30, color=color, stroke_width=2.6)
            name   = _txt(fn_name.upper(), color)
            name.next_to(swatch, RIGHT, buff=0.08)
            row = VGroup(swatch, name)

            if show_formula:
                fml = MathTex(formula_str, font_size=17).set_color(color)
                fml.next_to(row, DOWN, buff=0.08, aligned_edge=LEFT)
                row.add(fml)

            self._legend_rows.append(row)

        # stack legend on the right
        legend = VGroup(*self._legend_rows)
        legend.arrange(DOWN, buff=0.40, aligned_edge=LEFT)
        legend.next_to(self._axes, RIGHT, buff=0.38)
        self.add(legend)

        # axis labels
        x_lbl = _txt("x").next_to(self._axes, DOWN + RIGHT, buff=0.08)
        y_lbl = _txt("σ(x)").rotate(PI / 2).next_to(self._axes, LEFT, buff=0.38)
        self.add(x_lbl, y_lbl)

        # "negative → 0" annotation when ReLU is shown
        if "relu" in functions:
            note = _txt("negatives → 0", COLORS["ffn"], scale=LABEL_SCALE - 0.04)
            note.move_to(self._axes.c2p(-1.8, -0.4))
            self.add(note)

        # store function names for later lookups
        self._fn_names = list(functions)

    # ── animations ───────────────────────────────────────────────────────────

    def appear_animation(self, run_time: float = DEFAULT_RUN_TIME * 1.5) -> Succession:
        """Axes appear; curves draw in left-to-right, staggered."""
        return Succession(
            Create(self._axes, run_time=run_time * 0.35),
            LaggedStart(
                *[Create(c, run_time=run_time * 0.65) for c in self._curves],
                lag_ratio=0.30,
            ),
            LaggedStart(
                *[FadeIn(row) for row in self._legend_rows],
                lag_ratio=0.15, run_time=FAST_RUN_TIME,
            ),
        )

    def highlight_curve(
        self,
        fn_name:  str,
        run_time: float = FAST_RUN_TIME,
    ) -> AnimationGroup:
        """Brighten one named curve; dim all others."""
        anims = []
        for i, curve in enumerate(self._curves):
            if self._fn_names[i] == fn_name:
                anims.append(curve.animate(run_time=run_time)
                                  .set_stroke(width=3.8, opacity=1.0))
            else:
                anims.append(curve.animate(run_time=run_time)
                                  .set_stroke(width=1.0, opacity=0.18))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def reset_highlight(self, run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        return AnimationGroup(
            *[c.animate(run_time=run_time).set_stroke(width=2.6, opacity=1.0)
              for c in self._curves],
            lag_ratio=0.0,
        )

    def mark_x(
        self,
        x_val:    float,
        fn_name:  str   = "gelu",
        color:    str   = COLORS["highlight"],
        run_time: float = FAST_RUN_TIME,
    ) -> tuple[VGroup, Animation]:
        """
        Drop a dot + crosshair at (x_val, σ(x_val)) for `fn_name`.
        Returns (marker_group, animation).
        """
        fn, _ = _ACTIVATIONS.get(fn_name, _ACTIVATIONS["gelu"])
        y_val = fn(x_val)

        pt     = self._axes.c2p(x_val, y_val)
        dot    = Dot(pt, color=color, radius=0.09)
        h_line = DashedLine(self._axes.c2p(0, y_val), pt,
                            color=color, stroke_width=1.2)
        v_line = DashedLine(self._axes.c2p(x_val, -0.8), pt,
                            color=color, stroke_width=1.2)
        x_lbl  = _txt(f"{x_val:.1f}", color).next_to(
            self._axes.c2p(x_val, -0.8), DOWN, buff=0.06
        )
        y_lbl  = _txt(f"{y_val:.2f}", color).next_to(
            self._axes.c2p(0, y_val), LEFT, buff=0.06
        )
        group  = VGroup(dot, h_line, v_line, x_lbl, y_lbl)
        anim   = AnimationGroup(
            Create(h_line), Create(v_line),
            FadeIn(dot), Write(x_lbl), Write(y_lbl),
            lag_ratio=0.0, run_time=run_time,
        )
        return group, anim


# ===========================================================================
# 3.  NeuronFireAnim
# ===========================================================================

class NeuronFireAnim(VGroup):
    """
    A single neuron: an input value enters, an activation gate decides
    whether to fire, and an output value emerges.

    Layout (left → right):
        [input_val] ──wire──> ( neuron ) ──wire──> [output_val]
                                  ↑
                             fires / silent badge

    Parameters
    ----------
    input_val  : float — pre-activation scalar value
    activation : str   — 'relu' | 'gelu' | 'silu'
    neuron_r   : float — approximate radius of the neuron circle
    """

    def __init__(
        self,
        input_val:  float = 1.8,
        activation: str   = "gelu",
        neuron_r:   float = 0.36,
        **kwargs,
    ):
        super().__init__(**kwargs)

        fn, _ = _ACTIVATIONS.get(activation, _ACTIVATIONS["gelu"])
        output_val = fn(input_val)
        fired      = output_val > 1e-4

        body_color   = COLORS["ffn"]    if fired else COLORS["dim"]
        body_opacity = 0.88             if fired else 0.25
        wire_color   = COLORS["ffn"]    if fired else COLORS["dim"]
        status_str   = "FIRES"          if fired else "silent"
        status_color = COLORS["highlight"] if fired else GRAY

        # ── input label ───────────────────────────────────────────────────────
        in_lbl = MathTex(f"{input_val:.2f}", font_size=24).set_color(COLORS["embedding"])
        in_lbl.move_to(ORIGIN)

        # ── input wire ────────────────────────────────────────────────────────
        in_wire_start = in_lbl.get_right() + RIGHT * 0.10
        in_wire_end   = in_wire_start + RIGHT * 0.75
        in_wire = Line(in_wire_start, in_wire_end,
                       color=COLORS["embedding"], stroke_width=2.0)

        # ── neuron body ───────────────────────────────────────────────────────
        neuron = RoundedRectangle(
            width=neuron_r * 2.2, height=neuron_r * 2.2,
            corner_radius=neuron_r,
            fill_color=body_color, fill_opacity=body_opacity,
            stroke_color=WHITE, stroke_width=1.2,
        )
        neuron.next_to(in_wire, RIGHT, buff=0.05)

        act_icon = _txt(activation.upper(),
                        COLORS["query"] if fired else GRAY,
                        scale=LABEL_SCALE - 0.05)
        act_icon.move_to(neuron.get_center())

        # ── output wire ───────────────────────────────────────────────────────
        out_wire = Line(
            neuron.get_right() + RIGHT * 0.05,
            neuron.get_right() + RIGHT * 0.80,
            color=wire_color, stroke_width=2.0,
        )
        out_lbl = MathTex(f"{output_val:.3f}", font_size=24) \
                      .set_color(wire_color) \
                      .next_to(out_wire, RIGHT, buff=0.08)

        # ── status badge ──────────────────────────────────────────────────────
        badge_bg = RoundedRectangle(
            width=0.90, height=0.28, corner_radius=0.10,
            fill_color=status_color, fill_opacity=0.85 if fired else 0.30,
            stroke_width=0,
        )
        badge_txt = _txt(status_str, WHITE if fired else GRAY,
                         scale=LABEL_SCALE - 0.05)
        badge_txt.move_to(badge_bg.get_center())
        status_badge = VGroup(badge_bg, badge_txt)
        status_badge.next_to(neuron, UP, buff=0.16)

        # ── assemble ──────────────────────────────────────────────────────────
        self.add(in_lbl, in_wire, neuron, act_icon,
                 out_wire, out_lbl, status_badge)

        # stash refs for animation methods
        self._neuron       = neuron
        self._in_lbl       = in_lbl
        self._out_lbl      = out_lbl
        self._out_wire     = out_wire
        self._status_badge = status_badge
        self._fired        = fired
        self._body_color   = body_color

    # ── animations ───────────────────────────────────────────────────────────

    def build_animation(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """
        Sequential reveal:
          input value → wire grows → neuron fires (or dims) → output appears.
        """
        if self._fired:
            gate_anim = AnimationGroup(
                Flash(self._neuron, color=COLORS["highlight"], line_length=0.20),
                self._neuron.animate.set_fill(opacity=1.0),
                lag_ratio=0.0, run_time=run_time * 0.55,
            )
        else:
            gate_anim = self._neuron.animate(run_time=run_time * 0.30) \
                                    .set_fill(opacity=0.10)

        return Succession(
            FadeIn(self._in_lbl, run_time=FAST_RUN_TIME),
            Create(self._out_wire, run_time=run_time * 0.35),
            gate_anim,
            AnimationGroup(
                FadeIn(self._out_lbl, shift=RIGHT * 0.10),
                FadeIn(self._status_badge),
                lag_ratio=0.0, run_time=FAST_RUN_TIME,
            ),
        )

    def place_below(self, other: "NeuronFireAnim", buff: float = 0.42) -> "NeuronFireAnim":
        """Convenience: position this neuron below `other`."""
        self.next_to(other, DOWN, buff=buff)
        return self


# ===========================================================================
# 4.  DimensionExpansionAnim
# ===========================================================================

class DimensionExpansionAnim(VGroup):
    """
    Visualises d_model → d_ff → d_model with animated bars, arrows, and
    ratio label, optionally including the compression pass.

    Parameters
    ----------
    d_model        : int
    d_ff           : int
    show_compress  : bool — include the W₂ compression step
    """

    def __init__(
        self,
        d_model:      int  = 512,
        d_ff:         int  = 2048,
        show_compress: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_model      = d_model
        self.d_ff         = d_ff
        self._show_compress = show_compress

        bar_h    = 0.50
        w_narrow = 0.60
        w_wide   = 3.10
        gap      = 0.85

        # ── bars ─────────────────────────────────────────────────────────────
        self._in_bar  = _DimBar(d_model, w_narrow, bar_h,
                                COLORS["embedding"], f"d_model = {d_model}")
        self._hid_bar = _DimBar(d_ff,    w_wide,   bar_h,
                                COLORS["ffn"],      f"d_ff = {d_ff}")
        self._out_bar = _DimBar(d_model, w_narrow, bar_h,
                                COLORS["output"],   f"d_model = {d_model}")

        # ── expand / compress arrows ──────────────────────────────────────────
        self._in_bar.move_to(ORIGIN)
        self._hid_bar.move_to(
            _np.array([w_narrow / 2 + gap + w_wide / 2, 0, 0])
        )
        self._out_bar.move_to(
            _np.array([w_narrow / 2 + gap + w_wide + gap + w_narrow / 2, 0, 0])
        )

        exp_start = self._in_bar.get_right()
        exp_end   = self._hid_bar.get_left()
        cmp_start = self._hid_bar.get_right()
        cmp_end   = self._out_bar.get_left()

        self._exp_arr = Arrow(exp_start, exp_end, buff=0.08,
                              color=COLORS["ffn"], stroke_width=1.6,
                              max_tip_length_to_length_ratio=0.28)
        self._cmp_arr = Arrow(cmp_start, cmp_end, buff=0.08,
                              color=COLORS["value"], stroke_width=1.6,
                              max_tip_length_to_length_ratio=0.28)

        w1_lbl = _txt("W₁", COLORS["ffn"])
        w2_lbl = _txt("W₂", COLORS["value"])
        w1_lbl.next_to(self._exp_arr, UP, buff=0.08)
        w2_lbl.next_to(self._cmp_arr, UP, buff=0.08)

        self.add(self._in_bar, self._exp_arr, w1_lbl, self._hid_bar)
        if show_compress:
            self.add(self._cmp_arr, w2_lbl, self._out_bar)

        # ── ratio annotation ──────────────────────────────────────────────────
        ratio = d_ff // max(d_model, 1)
        ratio_lbl = MathTex(
            rf"d_{{ff}} = {ratio} \times d_{{model}}",
            font_size=22,
        ).set_color(COLORS["ffn"])
        ratio_lbl.next_to(self, UP, buff=0.32)
        self.add(ratio_lbl)

        # ── knowledge annotation ──────────────────────────────────────────────
        know_lbl = _txt(
            "FFN neurons encode factual knowledge",
            COLORS["highlight"],
            scale=LABEL_SCALE - 0.02,
        )
        know_lbl.next_to(self._hid_bar, DOWN, buff=0.32)
        self.add(know_lbl)

    # ── animations ───────────────────────────────────────────────────────────

    def build_animation(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """Input bar → expand arrow + hidden bar → compress arrow + output bar."""
        steps: list[Animation] = [
            self._in_bar.appear_animation(run_time),
            AnimationGroup(
                GrowArrow(self._exp_arr),
                self._hid_bar.appear_animation(run_time),
                lag_ratio=0.3,
            ),
        ]
        if self._show_compress:
            steps.append(
                AnimationGroup(
                    GrowArrow(self._cmp_arr),
                    self._out_bar.appear_animation(run_time),
                    lag_ratio=0.3,
                )
            )
        return Succession(*steps)

    def pulse_hidden(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """Flash + circumscribe the hidden bar to emphasise its size."""
        return Succession(
            Flash(self._hid_bar, color=COLORS["ffn"],
                  line_length=0.22, run_time=run_time * 0.45),
            Circumscribe(self._hid_bar, color=COLORS["ffn"], run_time=run_time),
        )

    def compare_ratio(
        self,
        new_d_ff:  int,
        run_time:  float = DEFAULT_RUN_TIME,
    ) -> tuple["DimensionExpansionAnim", FadeIn]:
        """
        Factory: create a second DimensionExpansionAnim with a different ratio
        and place it below for side-by-side comparison.
        Returns (new_anim, FadeIn animation).
        """
        other = DimensionExpansionAnim(
            d_model=self.d_model,
            d_ff=new_d_ff,
            show_compress=self._show_compress,
        )
        other.next_to(self, DOWN, buff=0.55)
        return other, FadeIn(other, run_time=run_time)


# ===========================================================================
# 5.  MoERouterAnim
# ===========================================================================

class MoERouterAnim(VGroup):
    """
    Mixture-of-Experts: input token → router (softmax) → top-k experts selected,
    rest dimmed.  Selected expert outputs are weighted-combined.

    Parameters
    ----------
    n_experts : int        — total number of expert FFNs
    top_k     : int        — number of experts activated per token (typically 1–2)
    selected  : list[int]  — which expert indices to select; random if None
    """

    def __init__(
        self,
        n_experts: int             = 8,
        top_k:     int             = 2,
        selected:  list[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_experts = n_experts
        self.top_k     = top_k

        if selected is None:
            rng      = _np.random.default_rng(42)
            selected = list(int(x) for x in rng.choice(n_experts, size=top_k, replace=False))
        self._selected = set(selected)

        expert_h   = 0.40
        expert_w   = 1.08
        expert_gap = 0.20
        total_h    = n_experts * expert_h + (n_experts - 1) * expert_gap

        # ── token box ─────────────────────────────────────────────────────────
        self._tok_box = _rounded_box("token  x", COLORS["token"], w=1.20, h=0.46)
        self._tok_box.move_to(ORIGIN)

        # ── router box ────────────────────────────────────────────────────────
        self._router = _rounded_box("Router\n(softmax)", COLORS["query"], w=1.40, h=0.58)
        self._router.next_to(self._tok_box, RIGHT, buff=0.90)

        self._tok_arr = Arrow(
            self._tok_box.get_right(), self._router.get_left(),
            buff=0.06, color=COLORS["arrow"], stroke_width=1.3,
            max_tip_length_to_length_ratio=0.30,
        )

        # ── expert boxes ──────────────────────────────────────────────────────
        experts_x = self._router.get_right()[0] + 2.00
        top_y     = total_h / 2 - expert_h / 2

        self._expert_boxes: list[VGroup] = []
        for k in range(n_experts):
            sel     = k in self._selected
            color   = COLORS["ffn"] if sel else COLORS["dim"]
            opacity = 0.88          if sel else 0.25
            box     = _rounded_box(f"Expert {k+1}", color,
                                   w=expert_w, h=expert_h, fill_opacity=opacity)
            box.move_to(_np.array([
                experts_x,
                top_y - k * (expert_h + expert_gap),
                0,
            ]))
            self._expert_boxes.append(box)
            self.add(box)

        # ── router → expert arrows ────────────────────────────────────────────
        self._router_arrows: list[Arrow] = []
        for k, box in enumerate(self._expert_boxes):
            sel   = k in self._selected
            color = COLORS["ffn"] if sel else COLORS["dim"]
            arr   = Arrow(
                self._router.get_right(), box.get_left(),
                buff=0.05, color=color,
                stroke_width=1.5 if sel else 0.55,
                stroke_opacity=1.0 if sel else 0.22,
                max_tip_length_to_length_ratio=0.28,
            )
            self._router_arrows.append(arr)
            self.add(arr)

        # ── probability bars (right of each expert box) ───────────────────────
        rng2 = _np.random.default_rng(77)
        raw  = _np.zeros(n_experts)
        for s in self._selected:
            raw[s] = rng2.uniform(1.8, 3.0)
        raw  += rng2.uniform(0.0, 0.4, n_experts)
        probs = _np.exp(raw) / _np.sum(_np.exp(raw))

        self._prob_bars: list[Rectangle] = []
        bar_max_w = 0.55
        for k, (box, p) in enumerate(zip(self._expert_boxes, probs)):
            bw  = max(0.04, float(p) * bar_max_w * n_experts * 0.38)
            bar = Rectangle(
                width=bw, height=expert_h * 0.42,
                fill_color=COLORS["ffn"] if k in self._selected else GRAY,
                fill_opacity=0.80 if k in self._selected else 0.22,
                stroke_width=0,
            )
            bar.next_to(box, RIGHT, buff=0.10)
            self._prob_bars.append(bar)
            self.add(bar)

            # percentage label
            pct = _txt(f"{p*100:.1f}%",
                       COLORS["ffn"] if k in self._selected else GRAY,
                       scale=LABEL_SCALE - 0.07)
            pct.next_to(bar, RIGHT, buff=0.06)
            self.add(pct)

        # ── weighted-combine output box ────────────────────────────────────────
        sel_boxes = [self._expert_boxes[k] for k in sorted(self._selected)]
        out_x     = (max(b.get_right()[0] for b in self._expert_boxes)
                     + max(b.width for b in self._prob_bars) + 1.20)
        mid_y     = float(_np.mean([b.get_center()[1] for b in sel_boxes]))

        self._combine_box = _rounded_box("Weighted\nCombine", COLORS["value"],
                                         w=1.30, h=0.56)
        self._combine_box.move_to(_np.array([out_x, mid_y, 0]))

        self._combine_arrows: list[Arrow] = []
        for k in sorted(self._selected):
            arr = Arrow(
                self._expert_boxes[k].get_right(),
                self._combine_box.get_left(),
                buff=0.06, color=COLORS["ffn"], stroke_width=1.3,
                max_tip_length_to_length_ratio=0.28,
            )
            self._combine_arrows.append(arr)
            self.add(arr)

        self.add(self._tok_box, self._router, self._tok_arr, self._combine_box)

        # ── title + top-k label ───────────────────────────────────────────────
        title = _txt(
            f"MoE  —  top-{top_k} of {n_experts} experts selected",
            COLORS["highlight"],
            scale=LABEL_SCALE + 0.02,
        )
        title.next_to(self, UP, buff=0.38)
        self.add(title)

        topk_tag = _txt(f"top-{top_k}", COLORS["ffn"])
        topk_tag.next_to(self._router, UP, buff=0.12)
        self.add(topk_tag)

    # ── animations ───────────────────────────────────────────────────────────

    def build_animation(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """
        Token appears → router → all arrows fan out (selected bright,
        rest dim) → experts appear → prob bars fill → selected flash →
        combine output.
        """
        fan_arrows = LaggedStart(
            *[GrowArrow(a) for a in self._router_arrows],
            lag_ratio=0.05, run_time=run_time * 0.55,
        )
        expert_appear = LaggedStart(
            *[FadeIn(b, shift=RIGHT * 0.08) for b in self._expert_boxes],
            lag_ratio=0.05, run_time=run_time * 0.45,
        )
        bar_appear = LaggedStart(
            *[FadeIn(b) for b in self._prob_bars],
            lag_ratio=0.04, run_time=run_time * 0.35,
        )
        flash_selected = AnimationGroup(
            *[Flash(self._expert_boxes[k], color=COLORS["highlight"],
                    line_length=0.20)
              for k in self._selected],
            lag_ratio=0.0, run_time=run_time * 0.45,
        )
        combine_anim = AnimationGroup(
            *[GrowArrow(a) for a in self._combine_arrows],
            FadeIn(self._combine_box),
            lag_ratio=0.0, run_time=run_time * 0.40,
        )
        return Succession(
            FadeIn(self._tok_box, run_time=FAST_RUN_TIME),
            GrowArrow(self._tok_arr),
            FadeIn(self._router, run_time=FAST_RUN_TIME),
            AnimationGroup(fan_arrows, expert_appear, lag_ratio=0.0),
            bar_appear,
            flash_selected,
            combine_anim,
        )

    def highlight_selected(self, run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        """Circumscribe the selected expert boxes."""
        return AnimationGroup(
            *[Circumscribe(self._expert_boxes[k],
                           color=COLORS["highlight"], run_time=run_time)
              for k in self._selected],
            lag_ratio=0.0,
        )

    def dim_unselected(self, run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        """Fade non-selected experts + their arrows to near-invisible."""
        anims = []
        for k in range(self.n_experts):
            if k not in self._selected:
                anims.append(
                    self._expert_boxes[k].animate(run_time=run_time).set_opacity(0.07)
                )
                anims.append(
                    self._router_arrows[k].animate(run_time=run_time).set_opacity(0.05)
                )
        return AnimationGroup(*anims, lag_ratio=0.0)

    def reset_opacity(self, run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        """Restore all experts to their original opacity."""
        anims = []
        for k, box in enumerate(self._expert_boxes):
            target_op = 0.88 if k in self._selected else 0.25
            anims.append(box.animate(run_time=run_time).set_opacity(target_op))
        return AnimationGroup(*anims, lag_ratio=0.0)