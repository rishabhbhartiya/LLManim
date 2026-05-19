"""
embeddings/positional_encoding.py
===================================
Manim assets for visualizing positional encodings in Transformers.

Assets
------
SineWaveDisplay     — Animated sine waves of varying frequency, one per dimension pair
PEMatrix            — Full (seq_len × d_model) positional-encoding heatmap
PositionBadges      — Token sequence with animated position-number badges
PEAdditionAnim      — Token-emb vector + PE vector → result, element-wise
RoPERotationAnim    — 2D rotation visualisation for Rotary Position Embedding

Design conventions (matches embedding_lookup.py)
-------------------------------------------------
  • Every class extends VGroup — composable, positionable, scalable
  • Every class exposes at least one *_animation() / build_animation() method
    returning a ready-to-play Manim Animation / Succession
  • Colors imported from styles.colors with graceful fallback
  • _np alias for numpy (avoids collision with Manim's re-exported np)

Usage
-----
    from embeddings.positional_encoding import PEMatrix, PEAdditionAnim

    class PEScene(Scene):
        def construct(self):
            mat = PEMatrix(seq_len=12, d_model=16)
            self.add(mat)
            self.play(mat.appear_animation())
            self.wait()
"""

from __future__ import annotations

from manim import (
    VGroup, Rectangle, Text, MathTex,
    Arrow, Line, Dot, Axes,
    Scene, Animation, AnimationGroup, Succession,
    FadeIn, FadeOut, Create, Write,
    Indicate, Flash,
    LaggedStart,
    GrowArrow, Rotate,
    LEFT, RIGHT, UP, DOWN, ORIGIN,
    WHITE, BLACK, GRAY, DARK_GRAY, LIGHT_GRAY,
    RED, GREEN, BLUE, YELLOW, ORANGE, TEAL,
    interpolate_color,
    ValueTracker,
    rate_functions, PI,
    Brace,
    NumberPlane, Vector,
    Arc,
    DashedLine, DashedVMobject,
)

import numpy as _np

# ---------------------------------------------------------------------------
# Shared style imports — graceful fallback so this file works in isolation
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
# Internal math helpers
# ---------------------------------------------------------------------------

def _pe_value(pos: int, dim: int, d_model: int) -> float:
    """
    Vaswani et al. 2017 sinusoidal positional encoding:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    i = dim // 2
    angle = pos / (10_000 ** (2 * i / max(d_model, 1)))
    return float(_np.sin(angle) if dim % 2 == 0 else _np.cos(angle))


def _pe_matrix(seq_len: int, d_model: int) -> _np.ndarray:
    """Return full PE array of shape (seq_len, d_model)."""
    mat = _np.zeros((seq_len, d_model))
    for p in range(seq_len):
        for d in range(d_model):
            mat[p, d] = _pe_value(p, d, d_model)
    return mat


def _value_to_color(v: float) -> str:
    """
    Map [-1, 1] → orange (negative) / dark-gray (zero) / teal (positive).
    Distinct palette from embedding_lookup so PE cells read differently.
    """
    t = (float(v) + 1.0) / 2.0          # clamp to [0, 1]
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return interpolate_color(ORANGE, DARK_GRAY, t * 2)
    else:
        return interpolate_color(DARK_GRAY, TEAL, (t - 0.5) * 2)


def _small_text(s: str, color=LIGHT_GRAY) -> Text:
    return Text(s, font="Monospace").scale(LABEL_SCALE).set_color(color)


# ===========================================================================
# 1.  SineWaveDisplay
# ===========================================================================

class SineWaveDisplay(VGroup):
    """
    Draws one sine + dashed-cosine pair per dimension pair, stacked vertically.
    Each row has a different frequency, making the varying-wavelength pattern
    visible at a glance.

    Parameters
    ----------
    d_model      : int   — total model dimension (frequencies computed from this)
    n_waves      : int   — how many dimension pairs to display (default 4)
    seq_len      : int   — x-axis range (number of positions to plot)
    show_formula : bool  — display the PE formula above the top wave
    axes_width   : float — width of each individual subplot
    axes_height  : float — height of each individual subplot
    """

    def __init__(
        self,
        d_model:      int   = 64,
        n_waves:      int   = 4,
        seq_len:      int   = 50,
        show_formula: bool  = True,
        axes_width:   float = 3.0,
        axes_height:  float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._axes_list: list[Axes]   = []
        self._curves:    list[VGroup] = []
        self._row_labels: list[Text]  = []

        wave_colors = [
            COLORS["embedding"], COLORS["query"],
            COLORS["value"],     COLORS["norm"],
        ]

        for idx in range(n_waves):
            i     = idx          # dimension pair index
            freq  = 1.0 / (10_000 ** (2 * i / max(d_model, 1)))
            color = wave_colors[idx % len(wave_colors)]

            ax = Axes(
                x_range=[0, seq_len, max(1, seq_len // 5)],
                y_range=[-1.3, 1.3, 0.5],
                x_length=axes_width,
                y_length=axes_height,
                axis_config={
                    "color":        COLORS["dim"],
                    "stroke_width": 0.9,
                    "include_tip":  False,
                },
                tips=False,
            )
            ax.shift(DOWN * idx * (axes_height + 0.5))
            self._axes_list.append(ax)

            # solid sine (even dim)
            sin_curve = ax.plot(
                lambda x, f=freq: _np.sin(f * x),
                x_range=[0, seq_len],
                color=color,
                stroke_width=2.2,
            )
            # dashed cosine (odd dim)
            cos_raw = ax.plot(
                lambda x, f=freq: _np.cos(f * x),
                x_range=[0, seq_len],
                color=color,
                stroke_width=1.2,
                stroke_opacity=0.5,
            )
            cos_curve = DashedVMobject(cos_raw, num_dashes=28)

            curve_group = VGroup(sin_curve, cos_curve)
            self._curves.append(curve_group)

            # row label on the left
            lbl = _small_text(f"i={i}", color=color)
            lbl.next_to(ax, LEFT, buff=0.18)
            self._row_labels.append(lbl)

            self.add(ax, curve_group, lbl)

        # legend: solid = sin, dashed = cos
        legend_sin = Line(ORIGIN, RIGHT * 0.35, color=WHITE, stroke_width=2.0)
        legend_sin_lbl = _small_text("sin  (even dim)").next_to(legend_sin, RIGHT, buff=0.08)
        legend_cos_raw = Line(ORIGIN, RIGHT * 0.35, color=WHITE, stroke_width=1.2, stroke_opacity=0.55)
        legend_cos = DashedVMobject(legend_cos_raw, num_dashes=5)
        legend_cos_lbl = _small_text("cos  (odd dim)").next_to(legend_cos, RIGHT, buff=0.08)
        legend_cos.next_to(VGroup(legend_sin, legend_sin_lbl), DOWN, buff=0.12)
        legend_cos_lbl.next_to(legend_cos, RIGHT, buff=0.08)
        legend = VGroup(legend_sin, legend_sin_lbl, legend_cos, legend_cos_lbl)
        legend.next_to(self._axes_list[-1], RIGHT, buff=0.3)
        self.add(legend)

        # optional formula
        if show_formula:
            formula = MathTex(
                r"\text{PE}(pos,2i)=\sin\!\left("
                r"\frac{pos}{10000^{2i/d}}\right)",
                font_size=20,
            ).set_color(LIGHT_GRAY).next_to(self._axes_list[0], UP, buff=0.28)
            self.add(formula)
            self._formula = formula
        else:
            self._formula = None

        # x-axis label beneath the bottom subplot
        x_lbl = _small_text("position  →")
        x_lbl.next_to(self._axes_list[-1], DOWN, buff=0.18)
        self.add(x_lbl)

    # ── animations ───────────────────────────────────────────────────────────

    def appear_animation(self, run_time: float = DEFAULT_RUN_TIME * 2) -> Succession:
        """Axes appear; then waves draw in top-to-bottom with labels."""
        axes_anim = LaggedStart(
            *[Create(ax) for ax in self._axes_list],
            lag_ratio=0.2, run_time=run_time * 0.35,
        )
        curve_anim = LaggedStart(
            *[Create(cg) for cg in self._curves],
            lag_ratio=0.3, run_time=run_time * 0.55,
        )
        label_anim = LaggedStart(
            *[FadeIn(lbl) for lbl in self._row_labels],
            lag_ratio=0.2, run_time=run_time * 0.3,
        )
        steps: list[Animation] = [axes_anim, AnimationGroup(curve_anim, label_anim)]
        if self._formula:
            steps.insert(0, Write(self._formula))
        return Succession(*steps)

    def highlight_wave(self, idx: int, run_time: float = 0.6) -> AnimationGroup:
        """Brighten one row, dim all others — useful for narrating each frequency."""
        anims = []
        for i, cg in enumerate(self._curves):
            opacity = 1.0 if i == idx else 0.15
            width   = 2.8  if i == idx else 1.2
            anims.append(cg.animate(run_time=run_time)
                           .set_stroke(opacity=opacity, width=width))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def reset_highlight(self, run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        anims = [
            cg.animate(run_time=run_time).set_stroke(opacity=1.0, width=2.0)
            for cg in self._curves
        ]
        return AnimationGroup(*anims, lag_ratio=0.0)

    def mark_position(
        self,
        pos:      int,
        wave_idx: int  = 0,
        color:    str  = COLORS["highlight"],
        run_time: float = FAST_RUN_TIME,
    ) -> tuple[VGroup, Animation]:
        """
        Drop a vertical dashed marker at `pos` on the wave at `wave_idx`.
        Returns (marker_group, Create animation).
        """
        ax   = self._axes_list[wave_idx]
        top  = ax.c2p(pos,  1.2)
        bot  = ax.c2p(pos, -1.2)
        line = DashedLine(bot, top, color=color, stroke_width=1.5)
        lbl  = _small_text(f"pos={pos}", color=color) \
                   .next_to(_np.array([top[0], top[1] + 0.08, 0]), UP, buff=0.04)
        group = VGroup(line, lbl)
        return group, Create(group, run_time=run_time)


# ===========================================================================
# 2.  PEMatrix
# ===========================================================================

class PEMatrix(VGroup):
    """
    Full (seq_len × d_model) sinusoidal PE heatmap.
    Rows = positions, columns = dimensions.
    Color encodes value on an orange → dark-gray → teal scale.

    Parameters
    ----------
    seq_len   : int   — number of position rows
    d_model   : int   — number of dimension columns
    cell_size : float — cell side length
    show_axes : bool  — annotate row/column indices and axis titles
    """

    def __init__(
        self,
        seq_len:   int   = 20,
        d_model:   int   = 32,
        cell_size: float = SMALL_CELL,
        show_axes: bool  = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.seq_len   = seq_len
        self.d_model   = d_model
        self.cell_size = cell_size
        self._pe       = _pe_matrix(seq_len, d_model)

        self._cells: list[list[Rectangle]] = []
        self._rows:  list[VGroup]          = []

        # ── grid ─────────────────────────────────────────────────────────────
        for r in range(seq_len):
            row_group = VGroup()
            cell_row  = []
            for c in range(d_model):
                v = self._pe[r, c]
                rect = Rectangle(
                    width=cell_size, height=cell_size,
                    fill_color=_value_to_color(v),
                    fill_opacity=0.88,
                    stroke_color=COLORS["dim"],
                    stroke_width=0.4,
                )
                rect.move_to(_np.array([c * cell_size, -r * cell_size, 0]))
                row_group.add(rect)
                cell_row.append(rect)
            self._cells.append(cell_row)
            self._rows.append(row_group)
            self.add(row_group)

        # ── axis annotations ─────────────────────────────────────────────────
        if show_axes:
            step_r = max(1, seq_len  // 6)
            step_c = max(1, d_model  // 6)

            for r in range(0, seq_len, step_r):
                lbl = _small_text(str(r)).next_to(self._cells[r][0], LEFT, buff=0.12)
                self.add(lbl)

            for c in range(0, d_model, step_c):
                lbl = _small_text(str(c)).next_to(self._cells[0][c], UP, buff=0.12)
                self.add(lbl)

            pos_title = _small_text("pos →") \
                            .rotate(PI / 2) \
                            .next_to(self, LEFT, buff=0.55)
            dim_title = _small_text("dim →").next_to(self, UP, buff=0.42)
            self.add(pos_title, dim_title)

        # ── title ─────────────────────────────────────────────────────────────
        title = Text("Sinusoidal PE Matrix", font="Monospace") \
                    .scale(LABEL_SCALE + 0.05) \
                    .set_color(COLORS["norm"]) \
                    .next_to(self, UP, buff=0.58)
        self.add(title)

    # ── animations ───────────────────────────────────────────────────────────

    def appear_animation(self, run_time: float = DEFAULT_RUN_TIME * 2.5) -> LaggedStart:
        """Rows materialize top-to-bottom; within each row cells appear l→r."""
        row_anims = [
            LaggedStart(
                *[FadeIn(cell, scale=0.75) for cell in row_group],
                lag_ratio=0.04,
                run_time=run_time / max(self.seq_len, 1) * 3.5,
            )
            for row_group in self._rows
        ]
        return LaggedStart(*row_anims, lag_ratio=0.07, run_time=run_time)

    def highlight_row(
        self,
        row_idx:  int,
        color:    str   = COLORS["highlight"],
        run_time: float = FAST_RUN_TIME,
    ) -> AnimationGroup:
        """Brighten one position row, dim all others."""
        anims = []
        for r, row in enumerate(self._rows):
            if r == row_idx:
                anims.append(row.animate(run_time=run_time)
                               .set_stroke(color=color, width=2.0)
                               .set_opacity(1.0))
            else:
                anims.append(row.animate(run_time=run_time).set_opacity(0.15))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def highlight_col(
        self,
        col_idx:  int,
        color:    str   = COLORS["embedding"],
        run_time: float = FAST_RUN_TIME,
    ) -> AnimationGroup:
        """Highlight one dimension column across all positions."""
        anims = []
        for r in range(self.seq_len):
            for c, cell in enumerate(self._cells[r]):
                if c == col_idx:
                    anims.append(cell.animate(run_time=run_time)
                                     .set_stroke(color=color, width=2.0)
                                     .set_opacity(1.0))
                else:
                    anims.append(cell.animate(run_time=run_time).set_opacity(0.12))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def reset_opacity(self, run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        """Restore all cells after a highlight."""
        return AnimationGroup(
            *[row.animate(run_time=run_time).set_opacity(0.88) for row in self._rows],
            lag_ratio=0.0,
        )

    def extract_row_vector(
        self,
        row_idx:         int,
        target_position = None,
        run_time:        float = DEFAULT_RUN_TIME,
    ) -> tuple[VGroup, Succession]:
        """
        Animate a copy of row `row_idx` sliding to `target_position`.
        Returns (row_copy, full_succession).
        """
        if target_position is None:
            target_position = self.get_bottom() + DOWN * 1.0

        row_copy = self._rows[row_idx].copy().set_opacity(1.0)
        anim = Succession(
            self.highlight_row(row_idx),
            row_copy.animate(
                run_time=run_time,
                rate_func=rate_functions.ease_in_out_sine,
            ).move_to(target_position),
        )
        return row_copy, anim


# ===========================================================================
# 3.  PositionBadges
# ===========================================================================

class PositionBadges(VGroup):
    """
    A row of token boxes, each with a 0-indexed position badge that
    appears one by one — ideal for introducing the concept of positional indices.

    Parameters
    ----------
    tokens      : list[str] — token strings to display
    badge_color : str       — fill color of the badge pill
    spacing     : float     — gap between successive token boxes
    """

    def __init__(
        self,
        tokens:      list[str] = None,
        badge_color: str       = COLORS["norm"],
        spacing:     float     = 0.22,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if tokens is None:
            tokens = ["[CLS]", "the", "cat", "sat", "on", "mat"]

        self._token_groups: list[VGroup] = []
        self._badges:       list[VGroup] = []

        tok_colors = [
            COLORS["token"],     COLORS["embedding"],
            COLORS["attention"], COLORS["ffn"],
            COLORS["query"],     COLORS["value"],
            COLORS["norm"],
        ]

        box_w, box_h = 0.88, 0.46
        x_cursor = 0.0

        for idx, tok in enumerate(tokens):
            color = tok_colors[idx % len(tok_colors)]

            # token rect
            rect = Rectangle(
                width=box_w, height=box_h,
                fill_color=color, fill_opacity=0.78,
                stroke_color=WHITE, stroke_width=1.0,
                corner_radius=0.07,
            )
            word = Text(tok, font="Monospace") \
                       .scale(0.30) \
                       .set_color(WHITE) \
                       .move_to(rect.get_center())
            token_group = VGroup(rect, word)
            token_group.move_to(_np.array([x_cursor, 0, 0]))

            # position badge (small pill below the token)
            pill = Rectangle(
                width=0.36, height=0.26,
                fill_color=badge_color, fill_opacity=0.92,
                stroke_color=WHITE, stroke_width=0.8,
                corner_radius=0.10,
            ).next_to(rect, DOWN, buff=0.10)
            num = Text(str(idx), font="Monospace") \
                      .scale(0.26) \
                      .set_color(WHITE) \
                      .move_to(pill.get_center())
            badge = VGroup(pill, num)
            badge.set_opacity(0)   # hidden until revealed

            self._token_groups.append(token_group)
            self._badges.append(badge)
            self.add(token_group, badge)
            x_cursor += box_w + spacing

        # caption under the badges
        caption = _small_text("position index", color=badge_color) \
                      .next_to(self, DOWN, buff=0.58)
        self.add(caption)

    # ── animations ───────────────────────────────────────────────────────────

    def appear_tokens(self, run_time: float = DEFAULT_RUN_TIME) -> LaggedStart:
        """Token boxes slide in from below, left-to-right."""
        return LaggedStart(
            *[FadeIn(tg, shift=UP * 0.12) for tg in self._token_groups],
            lag_ratio=0.14, run_time=run_time,
        )

    def reveal_badges(self, run_time: float = DEFAULT_RUN_TIME) -> LaggedStart:
        """Badges pop in one by one with a small flash."""
        return LaggedStart(
            *[
                AnimationGroup(
                    badge.animate.set_opacity(1.0),
                    Flash(badge, color=COLORS["highlight"], line_length=0.10),
                    lag_ratio=0.0,
                )
                for badge in self._badges
            ],
            lag_ratio=0.18,
            run_time=run_time,
        )

    def build_animation(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """Tokens appear, then badges drop in."""
        return Succession(self.appear_tokens(run_time), self.reveal_badges(run_time))

    def highlight_token(
        self,
        idx:      int,
        color:    str   = COLORS["highlight"],
        run_time: float = FAST_RUN_TIME,
    ) -> AnimationGroup:
        """Flash one token and its badge together."""
        return AnimationGroup(
            Indicate(self._token_groups[idx], color=color),
            Indicate(self._badges[idx],       color=color),
            lag_ratio=0.0, run_time=run_time,
        )


# ===========================================================================
# 4.  PEAdditionAnim
# ===========================================================================

class PEAdditionAnim(VGroup):
    """
    Vertical stacked layout:

        ┌─────────────────────────┐   ← token embedding  (green cells)
      +
        ┌─────────────────────────┐   ← PE row           (teal cells)
        ─────────────────────────
        ┌─────────────────────────┐   ← result           (blue cells)

    Parameters
    ----------
    pos      : int        — position index (selects PE row)
    tok_emb  : array-like — token embedding; random if None
    d_model  : int        — embedding dimension
    max_disp : int        — max cells shown (truncated with "…" beyond this)
    """

    def __init__(
        self,
        pos:      int  = 0,
        tok_emb        = None,
        d_model:  int  = 16,
        max_disp: int  = 12,
        **kwargs,
    ):
        super().__init__(**kwargs)

        rng = _np.random.default_rng(pos + 13)
        if tok_emb is None:
            tok_emb = rng.uniform(-0.8, 0.8, d_model)
        else:
            tok_emb = _np.array(tok_emb, dtype=float)[:d_model]

        pe_row = _np.array([_pe_value(pos, d, d_model) for d in range(d_model)])
        result = _np.clip(tok_emb + pe_row, -1.0, 1.0)

        # Build the three bars
        self._tok_bar = _VecBar(tok_emb, d_model, max_disp,
                                COLORS["token"],     f"Token Embedding")
        self._pe_bar  = _VecBar(pe_row,  d_model, max_disp,
                                COLORS["norm"],      f"PE  (pos={pos})")
        self._res_bar = _VecBar(result,  d_model, max_disp,
                                COLORS["embedding"], "Input to Layer 0")

        # Vertical layout
        self._tok_bar.move_to(ORIGIN)

        self._plus = MathTex("+").scale(1.0).set_color(WHITE)
        self._plus.next_to(self._tok_bar, DOWN, buff=0.25)

        self._pe_bar.next_to(self._plus, DOWN, buff=0.25)

        bar_w = self._tok_bar.width + 0.5
        self._sep = Line(LEFT * bar_w / 2, RIGHT * bar_w / 2,
                         color=GRAY, stroke_width=1.0)
        self._sep.next_to(self._pe_bar, DOWN, buff=0.22)

        self._res_bar.next_to(self._sep, DOWN, buff=0.22)

        self.add(self._tok_bar, self._plus, self._pe_bar, self._sep, self._res_bar)

    # ── animations ───────────────────────────────────────────────────────────

    def build_animation(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """
        1) Token embedding cells pop in  l→r
        2) Plus sign + PE cells appear
        3) Separator line draws
        4) Result cells rise in l→r
        """
        tok_in = self._tok_bar.appear_anim(run_time)
        pe_in  = AnimationGroup(
            FadeIn(self._plus), self._pe_bar.appear_anim(run_time),
            lag_ratio=0.0,
        )
        res_in = self._res_bar.appear_anim(run_time, shift_dir=UP)
        return Succession(tok_in, pe_in, Create(self._sep), res_in)

    def highlight_column(self, col_idx: int, run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        """
        Flash the same column across all three bars — narrate one dimension at a time.
        """
        anims = []
        for bar in (self._tok_bar, self._pe_bar, self._res_bar):
            for i, cell in enumerate(bar.cells):
                if i == col_idx:
                    anims.append(Indicate(cell, color=COLORS["highlight"], run_time=run_time))
                else:
                    anims.append(cell.animate(run_time=run_time).set_opacity(0.18))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def reset_opacity(self, run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        """Restore all cells."""
        anims = []
        for bar in (self._tok_bar, self._pe_bar, self._res_bar):
            for cell in bar.cells:
                anims.append(cell.animate(run_time=run_time).set_opacity(0.85))
        return AnimationGroup(*anims, lag_ratio=0.0)


# ---------------------------------------------------------------------------
# Internal helper for PEAdditionAnim — not exported
# ---------------------------------------------------------------------------

class _VecBar(VGroup):
    """Single labeled horizontal cell bar for a 1-D vector."""

    def __init__(
        self,
        values,
        d_model:     int,
        max_disp:    int,
        label_color: str,
        label_text:  str,
    ):
        super().__init__()
        cs = SMALL_CELL
        self.cells: list[Rectangle] = []
        n = min(len(values), max_disp)

        for i in range(n):
            v = float(values[i])
            rect = Rectangle(
                width=cs, height=cs * 1.65,
                fill_color=_value_to_color(v),
                fill_opacity=0.85,
                stroke_color=WHITE,
                stroke_width=0.5,
            )
            rect.move_to(_np.array([i * cs, 0, 0]))
            self.add(rect)
            self.cells.append(rect)

        if len(values) > max_disp:
            ell = Text("…", font="Monospace").scale(0.45).set_color(GRAY)
            ell.next_to(self.cells[-1], RIGHT, buff=0.08)
            self.add(ell)

        lbl = Text(label_text, font="Monospace") \
                  .scale(LABEL_SCALE) \
                  .set_color(label_color) \
                  .next_to(self, UP, buff=0.10)
        self.add(lbl)

    def appear_anim(
        self,
        run_time:  float = DEFAULT_RUN_TIME,
        shift_dir        = DOWN,
    ) -> LaggedStart:
        return LaggedStart(
            *[FadeIn(c, shift=shift_dir * 0.10) for c in self.cells],
            lag_ratio=0.06,
            run_time=run_time,
        )


# ===========================================================================
# 5.  RoPERotationAnim
# ===========================================================================

class RoPERotationAnim(VGroup):
    """
    Rotary Position Embedding (Su et al. 2021) visual.

    Renders a 2D unit-circle plane with one coloured arrow per requested
    position.  Each arrow is rotated by  θ_i · pos  where
    θ_i = 1 / 10000^(2i/d_model).

    The angle between any two arrows equals the *relative* angle, which is
    what makes RoPE capture relative position in the dot product.

    Parameters
    ----------
    d_model    : int        — model dimension
    dim_pair   : int        — dimension pair i to visualise (default 0 = slowest)
    positions  : list[int]  — up to 2 position indices (default [3, 7])
    plane_size : float      — side length of the NumberPlane (Manim units)
    """

    def __init__(
        self,
        d_model:    int       = 64,
        dim_pair:   int       = 0,
        positions:  list[int] = None,
        plane_size: float     = 3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if positions is None:
            positions = [3, 7]

        self._d_model   = d_model
        self._dim_pair  = dim_pair
        self._positions = list(positions[:2])
        self._freq      = 1.0 / (10_000 ** (2 * dim_pair / max(d_model, 1)))

        # ── plane ─────────────────────────────────────────────────────────────
        self._plane = NumberPlane(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=plane_size,
            y_length=plane_size,
            background_line_style={
                "stroke_color":   COLORS["dim"],
                "stroke_width":   0.6,
                "stroke_opacity": 0.35,
            },
            axis_config={"stroke_color": GRAY, "stroke_width": 0.9},
        )
        self.add(self._plane)

        # unit circle guide
        unit_circle = self._plane.plot_parametric_curve(
            lambda t: _np.array([_np.cos(t), _np.sin(t), 0]),
            t_range=[0, 2 * PI],
            color=COLORS["dim"],
            stroke_width=0.9,
            stroke_opacity=0.5,
        )
        self.add(unit_circle)

        # ── one arrow per position ────────────────────────────────────────────
        vec_colors = [COLORS["query"], COLORS["value"]]

        self._vectors:      list[Vector]  = []
        self._angle_arcs:   list[Arc]     = []
        self._angle_labels: list[MathTex] = []

        origin_pt = self._plane.c2p(0, 0)

        for k, pos in enumerate(self._positions):
            theta = pos * self._freq
            color = vec_colors[k % 2]

            tip_pt = self._plane.c2p(_np.cos(theta), _np.sin(theta))
            vec = Vector(
                direction=tip_pt - origin_pt,
                color=color,
                stroke_width=2.8,
            ).shift(origin_pt - self._plane.get_origin())

            # arc from 0 to theta
            arc_r = 0.30 * (plane_size / 3.0) * (k + 1)
            arc   = Arc(
                radius=arc_r,
                angle=theta,
                start_angle=0,
                arc_center=origin_pt,
                color=color,
                stroke_width=1.2,
            )

            # angle label placed at arc midpoint
            mid_angle = theta / 2
            lbl_r     = arc_r + 0.22
            lbl_world = self._plane.c2p(
                lbl_r * _np.cos(mid_angle),
                lbl_r * _np.sin(mid_angle),
            )
            angle_label = MathTex(
                rf"\theta_{{{dim_pair}}} \!\cdot\! {pos}",
                font_size=18,
            ).set_color(color).move_to(lbl_world)

            self._vectors.append(vec)
            self._angle_arcs.append(arc)
            self._angle_labels.append(angle_label)
            self.add(arc, vec, angle_label)

        # ── relative-angle label (only when 2 positions) ─────────────────────
        if len(self._positions) == 2:
            m, n = self._positions
            self._rel_label = MathTex(
                rf"\Delta\theta = ({m}-{n})\cdot\theta_{{{dim_pair}}}",
                font_size=19,
            ).set_color(COLORS["highlight"]) \
             .next_to(self._plane, DOWN, buff=0.32)
            self.add(self._rel_label)
        else:
            self._rel_label = None

        # ── key-insight label ─────────────────────────────────────────────────
        insight = Text(
            "relative angle encodes relative position",
            font="Monospace",
        ).scale(LABEL_SCALE - 0.04).set_color(LIGHT_GRAY)
        insight.next_to(self._plane, DOWN,
                        buff=0.65 if self._rel_label else 0.32)
        self.add(insight)

        # ── title ─────────────────────────────────────────────────────────────
        title = Text("RoPE — Rotary Positional Encoding", font="Monospace") \
                    .scale(LABEL_SCALE + 0.05) \
                    .set_color(COLORS["attention"]) \
                    .next_to(self, UP, buff=0.35)
        subtitle = _small_text(
            f"dim pair i={dim_pair}   "
            f"θ = {self._freq:.5f} rad/pos",
        ).next_to(title, DOWN, buff=0.05)
        self.add(title, subtitle)

    # ── animations ───────────────────────────────────────────────────────────

    def appear_animation(self, run_time: float = DEFAULT_RUN_TIME * 1.5) -> Succession:
        """Plane → vectors with arcs and labels, staggered."""
        vec_anims = LaggedStart(
            *[
                Succession(
                    GrowArrow(vec),
                    AnimationGroup(Create(arc), Write(lbl), lag_ratio=0.0),
                )
                for vec, arc, lbl in zip(
                    self._vectors, self._angle_arcs, self._angle_labels
                )
            ],
            lag_ratio=0.5,
            run_time=run_time * 0.7,
        )
        steps: list[Animation] = [Create(self._plane), vec_anims]
        if self._rel_label:
            steps.append(Write(self._rel_label))
        return Succession(*steps)

    def rotate_to_position(
        self,
        new_pos:  int,
        vec_idx:  int   = 0,
        run_time: float = DEFAULT_RUN_TIME,
    ) -> Rotate:
        """
        Continuously rotate vector `vec_idx` to the angle for `new_pos`.
        Use in a self.play() call to animate position scrubbing.
        """
        current_pos   = self._positions[vec_idx]
        delta_theta   = (new_pos - current_pos) * self._freq
        origin_pt     = self._plane.c2p(0, 0)
        self._positions[vec_idx] = new_pos
        return Rotate(
            self._vectors[vec_idx],
            angle=delta_theta,
            about_point=origin_pt,
            run_time=run_time,
        )

    def highlight_relative(self, run_time: float = 0.7) -> AnimationGroup:
        """Pulse the relative-angle label."""
        if self._rel_label is None:
            return AnimationGroup()
        return AnimationGroup(
            Flash(self._rel_label, color=COLORS["highlight"], line_length=0.22),
            Indicate(self._rel_label, color=COLORS["highlight"]),
            lag_ratio=0.0,
            run_time=run_time,
        )

    def side_by_side_comparison(
        self,
        other_dim_pair: int,
        run_time: float = DEFAULT_RUN_TIME,
    ) -> tuple["RoPERotationAnim", FadeIn]:
        """
        Factory: create a second RoPERotationAnim for a higher dim_pair
        (faster frequency) and place it to the right for comparison.
        Returns (new_anim_object, FadeIn_animation).
        """
        other = RoPERotationAnim(
            d_model=self._d_model,
            dim_pair=other_dim_pair,
            positions=self._positions,
        )
        other.next_to(self, RIGHT, buff=0.7)
        return other, FadeIn(other, run_time=run_time)