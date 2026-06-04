"""
llmanim/normalization/layer_norm.py
==============================================
Reusable Manim assets for the NORMALIZATION stage of a Transformer.

Classes
-------
    ValueDistributionBar   — horizontal strip of colored activation cells
    MeanSubtractionAnim    — shift every value so mean = 0
    VarianceScaleAnim      — divide by std-dev to unit variance
    GammaBetaAnim          — apply learnable scale (γ) and shift (β)
    LayerNormBlock         — full LayerNorm pipeline in one composable unit
    ResidualAnim           — skip connection curving around a sublayer
    PreVsPostNormCompare   — side-by-side Pre-Norm vs Post-Norm diagram
    RMSNormBlock           — RMSNorm variant (no mean subtraction; LLaMA-style)

Design principles
-----------------
  • Every class extends VGroup  →  composable, positionable, animatable.
  • Every ``animate_*`` method returns list[Animation], ready for
    ``self.play(*anims)``.
  • All magic numbers  →  styles/constants.py
  • All colors         →  styles/colors.py
  • All text           →  styles/fonts.py

Quick usage
-----------
    from manim import *
    from llmanim.normalization.layer_norm import (
        LayerNormBlock, ResidualAnim, PreVsPostNormCompare, RMSNormBlock,
    )

    class NormDemo(Scene):
        def construct(self):
            values = [2.3, -1.1, 0.5, 3.8, -0.7, 1.2, -2.9, 0.1]

            # Full LayerNorm walkthrough
            ln = LayerNormBlock(values)
            self.play(*ln.animate_full_pass())
            self.wait()

            # Residual connection around MHA
            res = ResidualAnim("MHA").shift(DOWN * 3)
            self.play(*res.animate_full())
            self.wait()
"""

from __future__ import annotations

import math

from manim import (
    VGroup, Rectangle, RoundedRectangle,
    Line, DashedLine, Arrow, CurvedArrow,
    Text, MathTex, Brace,
    FadeIn, FadeOut, Transform, Create,
    GrowFromEdge, AnimationGroup, Succession,
    UP, DOWN, LEFT, RIGHT,
    WHITE, ManimColor,
)

from llmanim.styles.colors import (
    UI_COLORS, COMPONENT_COLORS,
)
from llmanim.styles.constants import (
    TIMING, LAYOUT, NORM, MATRIX, TYPOGRAPHY as TY,
)
from llmanim.styles.fonts import (
    make_text, make_math, STYLE,
)


# ============================================================
# PURE-PYTHON MATH HELPERS  (no Manim dependency)
# ============================================================

def _mean(v: list[float]) -> float:
    return sum(v) / len(v)


def _std(v: list[float], mu: float, eps: float = 1e-5) -> float:
    return math.sqrt(sum((x - mu) ** 2 for x in v) / len(v) + eps)


def _rms(v: list[float], eps: float = 1e-5) -> float:
    return math.sqrt(sum(x ** 2 for x in v) / len(v) + eps)


def _ln(v: list[float]) -> list[float]:
    mu  = _mean(v)
    std = _std(v, mu)
    return [(x - mu) / std for x in v]


def _rms_norm(v: list[float]) -> list[float]:
    r = _rms(v)
    return [x / r for x in v]


def _broadcast(param: list[float] | float, n: int) -> list[float]:
    if isinstance(param, (int, float)):
        return [float(param)] * n
    return list(param)


# ============================================================
# COLOR HELPERS
# ============================================================

def _lerp_hex(a: str, b: str, t: float) -> str:
    """Linear-interpolate two hex colours, return hex string."""
    def h2r(h):
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r0, g0, b0 = h2r(a)
    r1, g1, b1 = h2r(b)
    return "#{:02X}{:02X}{:02X}".format(
        int(r0 + (r1 - r0) * t),
        int(g0 + (g1 - g0) * t),
        int(b0 + (b1 - b0) * t),
    )


# Raw hex constants so we avoid calling ._hex_string on ManimColor objects
_HEX_ZERO     = "#263238"   # matches UI_COLORS["zero"]
_HEX_POSITIVE = "#2196F3"   # matches UI_COLORS["positive"]
_HEX_NEGATIVE = "#F44336"   # matches UI_COLORS["negative"]
_HEX_NORM     = "#00BCD4"   # matches COMPONENT_COLORS["norm"]
_HEX_RESIDUAL = "#8BC34A"   # matches NORM.RESIDUAL_COLOR_HEX


def _cell_color(value: float, vmax: float = 4.0) -> ManimColor:
    """
    Map a float activation to a fill colour.
    negative → red intensity,  near-zero → dark,  positive → blue intensity.
    """
    t = max(-1.0, min(1.0, value / max(abs(vmax), 1e-9)))
    if t >= 0:
        return ManimColor(_lerp_hex(_HEX_ZERO, _HEX_POSITIVE, t))
    else:
        return ManimColor(_lerp_hex(_HEX_ZERO, _HEX_NEGATIVE, -t))


# ============================================================
# 1.  ValueDistributionBar
# ============================================================

class ValueDistributionBar(VGroup):
    """
    A horizontal row of colored cells representing a 1-D activation vector.

    Each cell's fill colour encodes sign and magnitude:
        positive  → blue    (darker = smaller)
        near-zero → dark grey
        negative  → red     (darker = smaller magnitude)

    Parameters
    ----------
    values : list[float]
        Activation values to display.
    cell_width : float
        Width of each cell in Manim units.
    cell_height : float
        Height of each cell in Manim units.
    show_values : bool
        Show numeric labels below each cell.
    label_text : str
        Optional heading label displayed to the left of the row.

    Key methods
    -----------
    animate_appear()              → list   Cells fade in left-to-right
    animate_update(new_values)    → list   Transform cells to new colours/labels
    animate_mean_line()           → list   Draw a dashed vertical mean indicator
    animate_std_brace()           → list   Draw a Brace showing std deviation
    get_values()                  → list[float]

    Example
    -------
    >>> bar = ValueDistributionBar([2.3, -1.1, 0.5, 3.8], label_text="x")
    >>> self.play(*bar.animate_appear())
    """

    def __init__(
        self,
        values: list[float],
        cell_width:  float = MATRIX.CELL_SIZE,
        cell_height: float = NORM.BAR_HEIGHT,
        show_values: bool  = True,
        label_text:  str   = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._values      = list(values)
        self._cell_w      = cell_width
        self._cell_h      = cell_height
        self._show_values = show_values
        self._n           = len(values)
        self._vmax        = max(abs(v) for v in values) if values else 1.0

        # ── Build cells ────────────────────────────────────────────────
        self._cells:      list[Rectangle] = []
        self._val_labels: list[Text]      = []

        for v in values:
            cell = Rectangle(
                width        = cell_width,
                height       = cell_height,
                fill_color   = _cell_color(v, self._vmax),
                fill_opacity = MATRIX.CELL_FILL_OPACITY,
                stroke_color = UI_COLORS["border"],
                stroke_width = MATRIX.CELL_STROKE,
            )
            self._cells.append(cell)

            if show_values:
                lbl = make_text(
                    f"{v:.2f}",
                    style = STYLE.MATRIX_VAL,
                    color = UI_COLORS["label_dim"],
                )
                self._val_labels.append(lbl)

        # Arrange in a row
        cell_row = VGroup(*self._cells)
        cell_row.arrange(RIGHT, buff=NORM.BAR_SPACING)

        # Value labels below their cells
        if show_values:
            for cell, lbl in zip(self._cells, self._val_labels):
                lbl.next_to(cell, DOWN, buff=0.08)

        # Optional heading label to the left
        self._heading: Text | None = None
        if label_text:
            self._heading = make_text(label_text, style=STYLE.LABEL_DIM)
            self._heading.next_to(cell_row, LEFT, buff=LAYOUT.MD)

        # Mean line / std brace built lazily in animate_* methods
        self._mean_line: DashedLine | None = None
        self._std_brace: Brace | None      = None

        self.add(cell_row)
        if show_values:
            self.add(*self._val_labels)
        if self._heading:
            self.add(self._heading)

    # ── Public helpers ─────────────────────────────────────────────────

    def get_values(self) -> list[float]:
        return list(self._values)

    def _set_values_silent(self, new_values: list[float]) -> None:
        self._values = list(new_values)
        self._vmax   = max(abs(v) for v in new_values) if new_values else 1.0

    # ── Animations ─────────────────────────────────────────────────────

    def animate_appear(self) -> list:
        """
        Cells (and heading) fade in left-to-right.

        Example
        -------
        >>> self.play(*bar.animate_appear())
        """
        anims: list = []
        if self._heading:
            anims.append(FadeIn(self._heading, run_time=TIMING.FAST))
        for i, cell in enumerate(self._cells):
            anims.append(FadeIn(cell, run_time=TIMING.TOKEN_APPEAR))
            if self._show_values:
                anims.append(FadeIn(self._val_labels[i], run_time=TIMING.TOKEN_APPEAR))
        return anims

    def animate_update(self, new_values: list[float]) -> list:
        """
        Transform every cell's colour and label to represent ``new_values``.

        Parameters
        ----------
        new_values : list[float]
            Must be the same length as the original values.

        Example
        -------
        >>> self.play(*bar.animate_update(centered_values), run_time=0.8)
        """
        assert len(new_values) == self._n, \
            f"animate_update: expected {self._n} values, got {len(new_values)}"

        vmax   = max(abs(v) for v in new_values) if new_values else 1.0
        anims: list = []

        for i, (cell, v) in enumerate(zip(self._cells, new_values)):
            new_cell = Rectangle(
                width        = self._cell_w,
                height       = self._cell_h,
                fill_color   = _cell_color(v, vmax),
                fill_opacity = MATRIX.CELL_FILL_OPACITY,
                stroke_color = UI_COLORS["border"],
                stroke_width = MATRIX.CELL_STROKE,
            )
            new_cell.move_to(cell.get_center())
            anims.append(Transform(cell, new_cell, run_time=TIMING.NORMAL))

            if self._show_values:
                new_lbl = make_text(
                    f"{v:.2f}",
                    style = STYLE.MATRIX_VAL,
                    color = UI_COLORS["label_dim"],
                )
                new_lbl.next_to(new_cell, DOWN, buff=0.08)
                anims.append(
                    Transform(self._val_labels[i], new_lbl, run_time=TIMING.NORMAL)
                )

        self._set_values_silent(new_values)
        return anims

    def animate_mean_line(self) -> list:
        """
        Draw a dashed vertical line at the mean-value x-position,
        with a μ = … label above it.

        Example
        -------
        >>> self.play(*bar.animate_mean_line())
        """
        mu       = _mean(self._values)
        row_cx   = sum(c.get_center()[0] for c in self._cells) / self._n
        y_top    = self._cells[0].get_top()[1]    + 0.25
        y_bot    = self._cells[0].get_bottom()[1] - 0.35

        self._mean_line = DashedLine(
            [row_cx, y_bot, 0],
            [row_cx, y_top, 0],
            color        = UI_COLORS["highlight"],
            stroke_width = 2.0,
            dash_length  = 0.10,
        )
        mean_lbl = make_text(
            f"μ = {mu:.2f}", style=STYLE.SMALL, color=UI_COLORS["highlight"]
        )
        mean_lbl.next_to(self._mean_line, UP, buff=0.08)
        self.add(self._mean_line, mean_lbl)
        return [
            Create(self._mean_line, run_time=TIMING.ARROW_DRAW),
            FadeIn(mean_lbl),
        ]

    def animate_std_brace(self) -> list:
        """
        Draw a Brace below the row with a σ = … label.

        Example
        -------
        >>> self.play(*bar.animate_std_brace())
        """
        mu  = _mean(self._values)
        std = _std(self._values, mu)
        self._std_brace = Brace(
            VGroup(*self._cells), DOWN, color=COMPONENT_COLORS["norm"]
        )
        brace_lbl = make_text(
            f"σ = {std:.2f}", style=STYLE.SMALL, color=COMPONENT_COLORS["norm"]
        )
        brace_lbl.next_to(self._std_brace, DOWN, buff=0.08)
        self.add(self._std_brace, brace_lbl)
        return [
            GrowFromEdge(self._std_brace, LEFT, run_time=TIMING.NORMAL),
            FadeIn(brace_lbl),
        ]


# ============================================================
# 2.  MeanSubtractionAnim
# ============================================================

class MeanSubtractionAnim(VGroup):
    """
    Visualise the mean-centring step of LayerNorm  (x → x − μ).

    Layout
    ------
    [Before bar]
         ↓   x_i − μ
    [After bar]

    Parameters
    ----------
    values : list[float]
        Raw activations before centring.

    Key methods
    -----------
    animate_show_mean()   → list   Appear before-bar + draw mean line
    animate_subtract()    → list   Transform after-bar to (x − μ) values
    animate_full()        → list   show_mean + subtract

    Example
    -------
    >>> ms = MeanSubtractionAnim([2.3, -1.1, 0.5, 3.8, -0.7])
    >>> self.add(ms)
    >>> self.play(*ms.animate_full())
    """

    def __init__(self, values: list[float], **kwargs):
        super().__init__(**kwargs)
        self._values   = list(values)
        self._mu       = _mean(values)
        self._centered = [v - self._mu for v in values]

        # Before (top) and After (bottom) bars
        self._before = ValueDistributionBar(values,          label_text="x")
        self._after  = ValueDistributionBar(values,          label_text="x − μ")
        self._after.next_to(self._before, DOWN, buff=1.0)

        # Connecting arrow
        self._arrow = Arrow(
            self._before.get_bottom() + DOWN * 0.08,
            self._after.get_top()     + UP   * 0.08,
            buff         = 0.0,
            color        = UI_COLORS["arrow"],
            stroke_width = 2.5,
            max_tip_length_to_length_ratio = 0.20,
        )

        # Formula alongside arrow
        self._fml = make_math(r"x_i - \mu", font_size=TY.FORMULA_SM)
        self._fml.next_to(self._arrow, RIGHT, buff=0.22)

        self.add(self._before, self._after, self._arrow, self._fml)

    # ── Animations ─────────────────────────────────────────────────────

    def animate_show_mean(self) -> list:
        """
        Reveal the before-bar and draw the mean indicator line.

        Example
        -------
        >>> self.play(*ms.animate_show_mean())
        """
        return self._before.animate_appear() + self._before.animate_mean_line()

    def animate_subtract(self) -> list:
        """
        Reveal the after-bar and transform its cells to (x − μ) values.

        Example
        -------
        >>> self.play(*ms.animate_subtract(), run_time=1.0)
        """
        anims  = [Create(self._arrow), FadeIn(self._fml)]
        anims += self._after.animate_appear()
        anims += self._after.animate_update(self._centered)
        anims += self._after.animate_mean_line()
        return anims

    def animate_full(self) -> list:
        """show_mean → subtract."""
        return self.animate_show_mean() + self.animate_subtract()

    # ── Output ─────────────────────────────────────────────────────────

    def get_centered(self) -> list[float]:
        return list(self._centered)


# ============================================================
# 3.  VarianceScaleAnim
# ============================================================

class VarianceScaleAnim(VGroup):
    """
    Visualise the variance-scaling step of LayerNorm  (x̂ → x̂ / σ).

    Layout
    ------
    [Centered bar]   ← std-dev brace below
          ↓   (x−μ) / σ
    [Normalised bar]

    Parameters
    ----------
    centered_values : list[float]
        Mean-centred activations (output of MeanSubtractionAnim).

    Key methods
    -----------
    animate_show_spread()   → list   Reveal centered bar + std brace
    animate_divide()        → list   Transform cells to (x−μ)/σ values
    animate_full()          → list   show_spread + divide

    Example
    -------
    >>> vs = VarianceScaleAnim(centered_values)
    >>> self.play(*vs.animate_full())
    """

    def __init__(self, centered_values: list[float], **kwargs):
        super().__init__(**kwargs)
        self._centered = list(centered_values)
        mu             = _mean(centered_values)
        self._std      = _std(centered_values, mu)
        self._normed   = [v / self._std for v in centered_values]

        self._before = ValueDistributionBar(centered_values, label_text="x − μ")
        self._after  = ValueDistributionBar(centered_values, label_text="(x−μ)/σ")
        self._after.next_to(self._before, DOWN, buff=1.0)

        self._arrow = Arrow(
            self._before.get_bottom() + DOWN * 0.08,
            self._after.get_top()     + UP   * 0.08,
            buff         = 0.0,
            color        = UI_COLORS["arrow"],
            stroke_width = 2.5,
            max_tip_length_to_length_ratio = 0.20,
        )
        self._fml = make_math(
            r"\frac{x_i - \mu}{\sigma + \varepsilon}", font_size=TY.FORMULA_SM
        )
        self._fml.next_to(self._arrow, RIGHT, buff=0.22)

        self.add(self._before, self._after, self._arrow, self._fml)

    # ── Animations ─────────────────────────────────────────────────────

    def animate_show_spread(self) -> list:
        """Reveal centered bar + std-deviation brace."""
        return self._before.animate_appear() + self._before.animate_std_brace()

    def animate_divide(self) -> list:
        """Draw connecting arrow + transform after-bar to (x−μ)/σ."""
        anims  = [Create(self._arrow), FadeIn(self._fml)]
        anims += self._after.animate_appear()
        anims += self._after.animate_update(self._normed)
        return anims

    def animate_full(self) -> list:
        return self.animate_show_spread() + self.animate_divide()

    def get_normed(self) -> list[float]:
        return list(self._normed)


# ============================================================
# 4.  GammaBetaAnim
# ============================================================

class GammaBetaAnim(VGroup):
    """
    Visualise the learnable affine step of LayerNorm  (γ·x̂ + β).

    Shows three rows: the normed vector, the γ parameter bar,
    the β parameter bar, and the final output vector.

    Parameters
    ----------
    normed_values : list[float]
        Unit-normalised activations (output of VarianceScaleAnim).
    gamma : list[float] | float
        Learnable scale parameter(s).  Scalar is broadcast.
    beta  : list[float] | float
        Learnable shift parameter(s).  Scalar is broadcast.

    Key methods
    -----------
    animate_show_params()   → list   Reveal γ and β bars
    animate_scale()         → list   Apply γ  (cells → γ·x̂)
    animate_shift()         → list   Apply β  (reveal final output)
    animate_full()          → list   show_params + scale + shift

    Example
    -------
    >>> gba = GammaBetaAnim(normed, gamma=1.2, beta=0.1)
    >>> self.play(*gba.animate_full())
    """

    def __init__(
        self,
        normed_values: list[float],
        gamma: list[float] | float = 1.0,
        beta:  list[float] | float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        n             = len(normed_values)
        gamma_list    = _broadcast(gamma, n)
        beta_list     = _broadcast(beta,  n)

        self._normed  = list(normed_values)
        self._gamma   = gamma_list
        self._beta    = beta_list
        self._scaled  = [g * x           for g, x    in zip(gamma_list, normed_values)]
        self._output  = [g * x + b       for g, x, b in zip(gamma_list, normed_values, beta_list)]

        SMALL_H = NORM.BAR_HEIGHT * 0.65    # thinner rows for γ and β

        # ── Bars ───────────────────────────────────────────────────────
        self._input_bar  = ValueDistributionBar(normed_values, label_text="x̂  (normed)")
        self._gamma_bar  = ValueDistributionBar(gamma_list, cell_height=SMALL_H, label_text="γ  (scale)")
        self._beta_bar   = ValueDistributionBar(beta_list,  cell_height=SMALL_H, label_text="β  (shift)")
        self._output_bar = ValueDistributionBar(self._output, label_text="γx̂ + β")

        self._gamma_bar.next_to(self._input_bar,  DOWN, buff=0.62)
        self._beta_bar.next_to( self._gamma_bar,  DOWN, buff=0.45)
        self._output_bar.next_to(self._beta_bar,  DOWN, buff=0.72)

        # ── Math labels for γ and β ────────────────────────────────────
        gamma_math = make_math(r"\gamma", font_size=TY.SUBHEADING)
        beta_math  = make_math(r"\beta",  font_size=TY.SUBHEADING)
        gamma_math.set_color(ManimColor("#FFEB3B"))
        beta_math.set_color( COMPONENT_COLORS["norm"])
        gamma_math.next_to(self._gamma_bar, LEFT, buff=0.12)
        beta_math.next_to( self._beta_bar,  LEFT, buff=0.12)

        # ── "Learnable" annotation ────────────────────────────────────
        learnable = make_text(
            "learnable parameters", style=STYLE.ANNOTATION
        )
        learnable.next_to(self._beta_bar, RIGHT, buff=0.38)

        # ── Full formula ───────────────────────────────────────────────
        self._fml = make_math(
            r"\gamma \hat{x} + \beta", font_size=TY.FORMULA
        )
        self._fml.next_to(self._output_bar, RIGHT, buff=0.45)

        self.add(
            self._input_bar,
            self._gamma_bar, gamma_math,
            self._beta_bar,  beta_math,
            learnable,
            self._output_bar,
            self._fml,
        )

    # ── Animations ─────────────────────────────────────────────────────

    def animate_show_params(self) -> list:
        """Reveal the input bar and both parameter bars."""
        anims  = self._input_bar.animate_appear()
        anims += self._gamma_bar.animate_appear()
        anims += self._beta_bar.animate_appear()
        return anims

    def animate_scale(self) -> list:
        """Transform input-bar cells to γ·x̂ values."""
        return self._input_bar.animate_update(self._scaled)

    def animate_shift(self) -> list:
        """Reveal output bar with final γx̂ + β values."""
        anims  = self._output_bar.animate_appear()
        anims += self._output_bar.animate_update(self._output)
        anims += [FadeIn(self._fml)]
        return anims

    def animate_full(self) -> list:
        """show_params → scale → shift."""
        return self.animate_show_params() + self.animate_scale() + self.animate_shift()

    def get_output(self) -> list[float]:
        return list(self._output)


# ============================================================
# 5.  LayerNormBlock
# ============================================================

class LayerNormBlock(VGroup):
    """
    Full LayerNorm pipeline in one composable asset.

    Chains:  input  →  MeanSubtractionAnim  →  VarianceScaleAnim
                    →  GammaBetaAnim  →  output

    Two rendering modes
    -------------------
    compact=False (default)
        Expands into the full step-by-step layout with formula header.
    compact=True
        Renders as a single rounded-rectangle labelled "Layer Norm",
        suitable for high-level block diagrams.

    Parameters
    ----------
    values : list[float]
        Raw input activations.
    gamma : list[float] | float
        Learnable scale (default 1.0).
    beta  : list[float] | float
        Learnable shift (default 0.0).
    compact : bool
        Compact single-box mode.

    Key methods
    -----------
    animate_full_pass()   → list   All steps sequentially (or compact appear)
    animate_compact()     → list   Single-box fade-in (compact mode only)
    get_output()          → list[float]

    Example
    -------
    >>> ln = LayerNormBlock([2.3, -1.1, 0.5, 3.8, -0.7, 1.2, -2.9, 0.1])
    >>> self.play(*ln.animate_full_pass())
    """

    def __init__(
        self,
        values: list[float],
        gamma:  list[float] | float = 1.0,
        beta:   list[float] | float = 0.0,
        compact: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        n          = len(values)
        gamma_list = _broadcast(gamma, n)
        beta_list  = _broadcast(beta,  n)

        self._compact = compact

        # Compute full pipeline values
        mu       = _mean(values)
        centered = [v - mu for v in values]
        std      = _std(centered, _mean(centered))
        normed   = [v / std for v in centered]
        output   = [g * x + b for g, x, b in zip(gamma_list, normed, beta_list)]
        self._output = output

        if compact:
            # ── Compact mode: single labeled box ──────────────────────
            self._box = RoundedRectangle(
                width          = 2.4,
                height         = 0.70,
                corner_radius  = 0.12,
                fill_color     = COMPONENT_COLORS["norm"],
                fill_opacity   = 0.25,
                stroke_color   = COMPONENT_COLORS["norm"],
                stroke_width   = TY.STROKE_NORMAL,
            )
            self._lbl = make_text(
                "Layer Norm", style=STYLE.COMPONENT_LABEL,
                color=COMPONENT_COLORS["norm"],
            )
            self._lbl.move_to(self._box.get_center())
            self.add(self._box, self._lbl)

        else:
            # ── Expanded mode: three-stage layout ─────────────────────

            # Stage objects
            self._step1 = MeanSubtractionAnim(values)
            self._step2 = VarianceScaleAnim(centered)
            self._step3 = GammaBetaAnim(normed, gamma_list, beta_list)

            # Lay out left → right with comfortable spacing
            self._step2.next_to(self._step1, RIGHT, buff=1.4)
            self._step3.next_to(self._step2, RIGHT, buff=1.4)

            # Inter-stage arrows
            def _stage_arrow(src, dst):
                return Arrow(
                    src.get_right() + RIGHT * 0.06,
                    dst.get_left()  + LEFT  * 0.06,
                    buff         = 0.0,
                    color        = UI_COLORS["arrow"],
                    stroke_width = 2.0,
                    max_tip_length_to_length_ratio = 0.20,
                )

            self._arr12 = _stage_arrow(self._step1, self._step2)
            self._arr23 = _stage_arrow(self._step2, self._step3)

            # Master formula at top of scene
            self._formula = make_math(
                r"\text{LN}(x) = \gamma \cdot"
                r"\frac{x - \mu}{\sigma + \varepsilon} + \beta",
                font_size=TY.FORMULA,
            )
            self._formula.to_edge(UP, buff=0.40)

            self.add(
                self._formula,
                self._step1,
                self._arr12,
                self._step2,
                self._arr23,
                self._step3,
            )

    # ── Output ─────────────────────────────────────────────────────────

    def get_output(self) -> list[float]:
        return list(self._output)

    # ── Animations ─────────────────────────────────────────────────────

    def animate_compact(self) -> list:
        """
        Fade in the compact single-box (compact=True mode only).

        Example
        -------
        >>> self.play(*ln_compact.animate_compact())
        """
        if not self._compact:
            return [FadeIn(self)]
        return [FadeIn(self._box), FadeIn(self._lbl)]

    def animate_full_pass(self) -> list:
        """
        Step through the full LayerNorm pipeline in order:
          formula → mean subtraction → variance scaling → γ/β affine.

        In compact mode, behaves like animate_compact().

        Example
        -------
        >>> self.play(*ln.animate_full_pass())
        """
        if self._compact:
            return self.animate_compact()

        anims: list = []
        anims += [FadeIn(self._formula, run_time=TIMING.LABEL_FADE)]
        anims += self._step1.animate_full()
        anims += [Create(self._arr12, run_time=TIMING.ARROW_DRAW)]
        anims += self._step2.animate_full()
        anims += [Create(self._arr23, run_time=TIMING.ARROW_DRAW)]
        anims += self._step3.animate_full()
        return anims


# ============================================================
# 6.  ResidualAnim
# ============================================================

class ResidualAnim(VGroup):
    """
    Visualise a residual (skip) connection around a Transformer sublayer.

    Layout
    ------
    ──── x ────┬──[ Sublayer ]──── F(x) ────┬──(+)──── x + F(x) ──→
               │                             │
               └─────────── skip ────────────┘

    The skip arc curves *below* the sublayer box (``angle=-1.1``).

    Parameters
    ----------
    sublayer_label : str
        Text inside the sublayer box (e.g. "MHA", "FFN").
    sublayer_color : ManimColor, optional
        Accent colour for the box border and label.
    show_formula : bool
        Append "x + F(x)" formula near the Add node.

    Key methods
    -----------
    animate_input()      → list   Input arrow + 'x' label appear
    animate_sublayer()   → list   Sublayer box fades in
    animate_skip()       → list   Skip arc draws around the box
    animate_add()        → list   (+) node + output arrow appear
    animate_full()       → list   All four steps in sequence

    Example
    -------
    >>> res = ResidualAnim("MHA", sublayer_color=COMPONENT_COLORS["attention"])
    >>> self.play(*res.animate_full())
    """

    def __init__(
        self,
        sublayer_label: str = "Sublayer",
        sublayer_color: ManimColor | None = None,
        show_formula:   bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        col = sublayer_color or COMPONENT_COLORS["norm"]

        # ── Sublayer box ───────────────────────────────────────────────
        self._box = RoundedRectangle(
            width          = 2.8,
            height         = 0.80,
            corner_radius  = 0.14,
            fill_color     = col,
            fill_opacity   = 0.18,
            stroke_color   = col,
            stroke_width   = TY.STROKE_NORMAL,
        )
        self._box_lbl = make_text(
            sublayer_label, style=STYLE.COMPONENT_LABEL, color=col
        )
        self._box_lbl.move_to(self._box.get_center())

        # ── Input arrow ────────────────────────────────────────────────
        in_start  = self._box.get_left()  + LEFT  * 1.6
        in_end    = self._box.get_left()

        self._input_arrow = Arrow(
            in_start, in_end,
            buff         = 0.08,
            color        = UI_COLORS["arrow"],
            stroke_width = 2.5,
            max_tip_length_to_length_ratio = 0.20,
        )
        self._input_lbl = make_text("x", style=STYLE.LABEL, color=UI_COLORS["label"])
        self._input_lbl.next_to(self._input_arrow, UP, buff=0.10)

        # ── Through arrow (box → add node) ─────────────────────────────
        self._through_arrow = Arrow(
            self._box.get_right(),
            self._box.get_right() + RIGHT * 1.1,
            buff         = 0.08,
            color        = UI_COLORS["arrow"],
            stroke_width = 2.5,
            max_tip_length_to_length_ratio = 0.20,
        )
        fx_lbl = make_text("F(x)", style=STYLE.LABEL_DIM, color=UI_COLORS["label_dim"])
        fx_lbl.next_to(self._through_arrow, UP, buff=0.10)

        # ── Add (+) node ───────────────────────────────────────────────
        add_pos = self._through_arrow.get_end()
        self._add_circle = RoundedRectangle(
            width          = 0.46,
            height         = 0.46,
            corner_radius  = 0.23,
            fill_color     = ManimColor(_HEX_NORM),
            fill_opacity   = 0.35,
            stroke_color   = ManimColor(_HEX_NORM),
            stroke_width   = TY.STROKE_NORMAL,
        )
        self._add_circle.move_to(add_pos)
        self._plus = make_text("+", style=STYLE.HEADING, color=ManimColor(_HEX_NORM))
        self._plus.move_to(add_pos)

        # ── Output arrow ───────────────────────────────────────────────
        self._out_arrow = Arrow(
            add_pos + RIGHT * 0.24,
            add_pos + RIGHT * 1.5,
            buff         = 0.0,
            color        = UI_COLORS["arrow"],
            stroke_width = 2.5,
            max_tip_length_to_length_ratio = 0.20,
        )
        out_lbl = make_text("x + F(x)", style=STYLE.LABEL, color=UI_COLORS["label"])
        out_lbl.next_to(self._out_arrow, UP, buff=0.10)

        # ── Skip arc (curves below the sublayer) ───────────────────────
        skip_start = in_start + RIGHT * 0.35
        skip_end   = add_pos  + DOWN  * 0.24

        self._skip_arc = CurvedArrow(
            skip_start, skip_end,
            angle        = -1.10,          # bows downward
            color        = ManimColor(_HEX_RESIDUAL),
            stroke_width = 2.5,
        )
        skip_lbl = make_text("skip", style=STYLE.ANNOTATION,
                              color=ManimColor(_HEX_RESIDUAL))
        skip_lbl.next_to(self._skip_arc, DOWN, buff=0.08)

        # ── Optional formula ───────────────────────────────────────────
        self._fml: MathTex | None = None
        if show_formula:
            self._fml = make_math(
                r"\text{output} = x + F(x)", font_size=TY.FORMULA_SM
            )
            self._fml.next_to(self._add_circle, UP, buff=0.60)

        self.add(
            self._input_arrow, self._input_lbl,
            self._box, self._box_lbl,
            self._through_arrow, fx_lbl,
            self._add_circle, self._plus,
            self._out_arrow, out_lbl,
            self._skip_arc, skip_lbl,
        )
        if self._fml is not None:
            self.add(self._fml)

    # ── Animations ─────────────────────────────────────────────────────

    def animate_input(self) -> list:
        """Draw the input arrow and 'x' label."""
        return [
            Create(self._input_arrow, run_time=TIMING.ARROW_DRAW),
            FadeIn(self._input_lbl),
        ]

    def animate_sublayer(self) -> list:
        """Fade in the sublayer box."""
        return [FadeIn(self._box), FadeIn(self._box_lbl)]

    def animate_skip(self) -> list:
        """Draw the skip arc from the input branch to the Add node."""
        return [Create(self._skip_arc, run_time=TIMING.NORMAL)]

    def animate_add(self) -> list:
        """Draw through-arrow, Add node, output arrow, and formula."""
        anims = [
            Create(self._through_arrow, run_time=TIMING.ARROW_DRAW),
            FadeIn(self._add_circle),
            FadeIn(self._plus),
            Create(self._out_arrow, run_time=TIMING.ARROW_DRAW),
        ]
        if self._fml is not None:
            anims.append(FadeIn(self._fml, run_time=TIMING.LABEL_FADE))
        return anims

    def animate_full(self) -> list:
        """input → sublayer → skip arc → add node + output."""
        return (
            self.animate_input()
            + self.animate_sublayer()
            + self.animate_skip()
            + self.animate_add()
        )


# ============================================================
# 7.  PreVsPostNormCompare
# ============================================================

class PreVsPostNormCompare(VGroup):
    """
    Side-by-side diagram contrasting Pre-Norm (modern) and Post-Norm (original).

    Pre-Norm   (GPT-2, LLaMA, Mistral …):
        x  →  LN  →  Sublayer  →  Add  →  output

    Post-Norm  (original "Attention is All You Need"):
        x  →  Sublayer  →  Add  →  LN  →  output

    Parameters
    ----------
    sublayer_label : str
        Label shown inside the sublayer box in both diagrams.

    Key methods
    -----------
    animate_left()    → list   Reveal Pre-Norm diagram
    animate_right()   → list   Reveal Post-Norm diagram
    animate_full()    → list   Both diagrams + explanatory note

    Example
    -------
    >>> cmp = PreVsPostNormCompare("FFN")
    >>> self.play(*cmp.animate_full())
    """

    def __init__(self, sublayer_label: str = "Sublayer", **kwargs):
        super().__init__(**kwargs)

        # ── Pre-Norm (left) ────────────────────────────────────────────
        pre_title = make_text(
            "Pre-Norm  (modern)", style=STYLE.HEADING, color=COMPONENT_COLORS["norm"]
        )
        pre_sub = make_text(
            "LN → Sublayer → Add", style=STYLE.CAPTION, color=UI_COLORS["label_dim"]
        )
        pre_sub.next_to(pre_title, DOWN, buff=0.12)

        pre_flow = self._build_flow(
            labels        = ["x", "LN", sublayer_label, "+", "out"],
            highlight_idx = [1],
            color         = COMPONENT_COLORS["norm"],
        )
        pre_flow.next_to(pre_sub, DOWN, buff=0.45)
        self._pre_group = VGroup(pre_title, pre_sub, pre_flow)

        # ── Post-Norm (right) ──────────────────────────────────────────
        post_title = make_text(
            "Post-Norm  (original)", style=STYLE.HEADING, color=UI_COLORS["label_dim"]
        )
        post_sub = make_text(
            "Sublayer → Add → LN", style=STYLE.CAPTION, color=UI_COLORS["label_dim"]
        )
        post_sub.next_to(post_title, DOWN, buff=0.12)

        post_flow = self._build_flow(
            labels        = ["x", sublayer_label, "+", "LN", "out"],
            highlight_idx = [3],
            color         = UI_COLORS["label_dim"],
        )
        post_flow.next_to(post_sub, DOWN, buff=0.45)
        self._post_group = VGroup(post_title, post_sub, post_flow)
        self._post_group.next_to(self._pre_group, RIGHT, buff=1.8)

        # ── Vertical divider ───────────────────────────────────────────
        mid_x = (
            self._pre_group.get_right()[0] + self._post_group.get_left()[0]
        ) / 2
        divider = DashedLine(
            [mid_x,  2.2, 0],
            [mid_x, -2.2, 0],
            color        = UI_COLORS["grid_line"],
            stroke_width = 1.2,
            dash_length  = 0.12,
        )

        # ── Explanatory note ───────────────────────────────────────────
        self._note = make_text(
            "Pre-Norm stabilises training; widely adopted post-2019.",
            style=STYLE.ANNOTATION,
        )
        self._note.next_to(
            VGroup(self._pre_group, self._post_group), DOWN, buff=0.55
        )

        self.add(self._pre_group, self._post_group, divider, self._note)

    # ── Internal builder ───────────────────────────────────────────────

    @staticmethod
    def _build_flow(
        labels: list[str],
        highlight_idx: list[int],
        color: ManimColor,
    ) -> VGroup:
        """Horizontal chain of labeled boxes connected by arrows."""
        boxes: list[VGroup] = []

        for i, lbl in enumerate(labels):
            hl = (i in highlight_idx)
            box = RoundedRectangle(
                width          = 1.05,
                height         = 0.48,
                corner_radius  = 0.10,
                fill_color     = color if hl else UI_COLORS["surface"],
                fill_opacity   = 0.60  if hl else 0.14,
                stroke_color   = color if hl else UI_COLORS["border"],
                stroke_width   = TY.STROKE_NORMAL if hl else TY.STROKE_THIN,
            )
            txt = make_text(
                lbl,
                style = STYLE.LABEL     if hl else STYLE.LABEL_DIM,
                color = color           if hl else UI_COLORS["label_dim"],
            )
            txt.move_to(box.get_center())
            boxes.append(VGroup(box, txt))

        row = VGroup(*boxes)
        row.arrange(RIGHT, buff=0.28)

        arrows = VGroup(*[
            Arrow(
                boxes[i].get_right(),
                boxes[i + 1].get_left(),
                buff         = 0.05,
                color        = UI_COLORS["arrow_dim"],
                stroke_width = 1.8,
                max_tip_length_to_length_ratio = 0.25,
            )
            for i in range(len(boxes) - 1)
        ])

        return VGroup(row, arrows)

    # ── Animations ─────────────────────────────────────────────────────

    def animate_left(self) -> list:
        """Reveal the Pre-Norm diagram."""
        return [FadeIn(self._pre_group, run_time=TIMING.NORMAL)]

    def animate_right(self) -> list:
        """Reveal the Post-Norm diagram."""
        return [FadeIn(self._post_group, run_time=TIMING.NORMAL)]

    def animate_full(self) -> list:
        """Left diagram → right diagram → explanatory note."""
        return (
            self.animate_left()
            + self.animate_right()
            + [FadeIn(self._note, run_time=TIMING.SLOW)]
        )


# ============================================================
# 8.  RMSNormBlock
# ============================================================

class RMSNormBlock(VGroup):
    """
    RMSNorm — the LayerNorm variant used in LLaMA, Mistral, Falcon, and most
    modern open-weight models.

    Difference from LayerNorm
    -------------------------
    • Skips mean subtraction entirely (no centering step).
    • Divides by the root-mean-square instead of std dev.
    • Has only γ (scale); no β (shift).

    Pipeline:  x  →  x / RMS(x)  →  γ · x̂

    Parameters
    ----------
    values : list[float]
        Raw input activations.
    gamma : list[float] | float
        Learnable scale (default 1.0).
    compact : bool
        Compact single-box mode for high-level diagrams.

    Key methods
    -----------
    animate_full_pass()   → list   Full pipeline (or compact appear)
    animate_compact()     → list   Single-box fade-in (compact mode only)
    get_output()          → list[float]

    Example
    -------
    >>> rms = RMSNormBlock([2.3, -1.1, 0.5, 3.8, -0.7, 1.2, -2.9, 0.1])
    >>> self.play(*rms.animate_full_pass())
    """

    def __init__(
        self,
        values: list[float],
        gamma:  list[float] | float = 1.0,
        compact: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        n          = len(values)
        gamma_list = _broadcast(gamma, n)

        self._compact = compact

        rms_val = _rms(values)
        normed  = _rms_norm(values)
        output  = [g * x for g, x in zip(gamma_list, normed)]
        self._output = output

        if compact:
            # ── Compact box ────────────────────────────────────────────
            self._box = RoundedRectangle(
                width          = 2.6,
                height         = 0.70,
                corner_radius  = 0.12,
                fill_color     = COMPONENT_COLORS["norm"],
                fill_opacity   = 0.25,
                stroke_color   = COMPONENT_COLORS["norm"],
                stroke_width   = TY.STROKE_NORMAL,
            )
            self._lbl = make_text(
                "RMS Norm", style=STYLE.COMPONENT_LABEL,
                color=COMPONENT_COLORS["norm"],
            )
            self._lbl.move_to(self._box.get_center())
            self.add(self._box, self._lbl)

        else:
            # ── Expanded layout ────────────────────────────────────────
            self._input_bar  = ValueDistributionBar(values,  label_text="x")
            self._normed_bar = ValueDistributionBar(normed,  label_text="x / RMS(x)")
            self._output_bar = ValueDistributionBar(output,  label_text="γ · x̂")

            self._normed_bar.next_to(self._input_bar,  DOWN, buff=0.90)
            self._output_bar.next_to(self._normed_bar, DOWN, buff=0.90)

            def _arr(src, dst):
                return Arrow(
                    src.get_bottom() + DOWN * 0.08,
                    dst.get_top()    + UP   * 0.08,
                    buff         = 0.0,
                    color        = UI_COLORS["arrow"],
                    stroke_width = 2.0,
                    max_tip_length_to_length_ratio = 0.20,
                )

            arr1 = _arr(self._input_bar,  self._normed_bar)
            arr2 = _arr(self._normed_bar, self._output_bar)

            fml1 = make_math(
                r"\frac{x_i}{\text{RMS}(x) + \varepsilon}", font_size=TY.FORMULA_SM
            )
            fml2 = make_math(r"\gamma \hat{x}", font_size=TY.FORMULA_SM)
            fml1.next_to(arr1, RIGHT, buff=0.20)
            fml2.next_to(arr2, RIGHT, buff=0.20)

            rms_badge = make_text(
                f"RMS = {rms_val:.3f}", style=STYLE.CAPTION,
                color=UI_COLORS["highlight"],
            )
            rms_badge.next_to(self._input_bar, RIGHT, buff=0.50)

            title = make_math(
                r"\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma",
                font_size=TY.FORMULA,
            )
            title.to_edge(UP, buff=0.40)

            self.add(
                title,
                self._input_bar, rms_badge,
                arr1, fml1,
                self._normed_bar,
                arr2, fml2,
                self._output_bar,
            )

    # ── Output ─────────────────────────────────────────────────────────

    def get_output(self) -> list[float]:
        return list(self._output)

    # ── Animations ─────────────────────────────────────────────────────

    def animate_compact(self) -> list:
        """Fade in compact box (compact=True mode only)."""
        if not self._compact:
            return [FadeIn(self)]
        return [FadeIn(self._box), FadeIn(self._lbl)]

    def animate_full_pass(self) -> list:
        """
        Step through RMSNorm:
          title → input bar → divide by RMS → γ scale → output.

        Example
        -------
        >>> self.play(*rms.animate_full_pass())
        """
        if self._compact:
            return self.animate_compact()

        anims  = self._input_bar.animate_appear()
        anims += self._normed_bar.animate_appear()
        anims += self._normed_bar.animate_update(
            [v for v in self._normed_bar.get_values()]
        )
        anims += self._output_bar.animate_appear()
        return anims