"""
attention/attention_score.py
────────────────────────────
Reusable Manim assets for the Attention Score computation stage.

  QKᵀ  →  scale by 1/√dₖ  →  (optional causal mask)  →  Softmax  →  Weights

Assets
------
DotProductAnim      — Q[i] · K[j] → scalar score, element-wise then sum
ScoreMatrix         — seq_len × seq_len grid of raw scores, filled cell by cell
ScalingAnim         — divides every score by √dₖ, before/after comparison
CausalMask          — upper-triangle -∞ overlay with animated reveal
SoftmaxAnim         — raw score row → probability row with curve & formula
AttentionHeatmap    — final weight matrix as a labeled heatmap

Usage
-----
    from attention.attention_score import (
        DotProductAnim, ScoreMatrix, ScalingAnim,
        CausalMask, SoftmaxAnim, AttentionHeatmap,
    )

Design Contract (same as attention_output.py)
---------------------------------------------
  • Every class is a VGroup — move/scale/transform freely.
  • build_anim() → Succession  plays the full sequence.
  • Fine-grained helpers for composition.
  • Graceful fallback if styles/colors.py is absent.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from manim import (
    RIGHT, LEFT, UP, DOWN, ORIGIN,
    WHITE, BLACK, GRAY, YELLOW, RED, BLUE,
    AnimationGroup, FadeIn, FadeOut, GrowArrow,
    LaggedStart, Succession, Transform,
    VGroup, VMobject,
    Rectangle, RoundedRectangle, Arrow, Line, DashedLine,
    Text, MathTex, Tex,
    DecimalNumber, Integer,
    Brace,
    Scene,
    Create, Write, Indicate, Flash,
    Dot, Axes, FunctionGraph,
    CurvedArrow,
    SurroundingRectangle,
    config,
    DEGREES,
)
from manim import ManimColor, interpolate_color

# ─────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────
try:
    from styles.colors import COLORS  # type: ignore
except ImportError:
    COLORS: dict = {
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
        "mask":         "#1A1A2E",
    }

def _hex(key: str) -> ManimColor:
    return ManimColor(COLORS.get(key, "#FFFFFF"))

def _lerp(c1: str, c2: str, t: float) -> ManimColor:
    return interpolate_color(ManimColor(c1), ManimColor(c2), np.clip(t, 0, 1))

def _score_color(score: float, lo: float = -4.0, hi: float = 4.0) -> ManimColor:
    """Low score → dim purple, high score → bright yellow."""
    t = (score - lo) / (hi - lo + 1e-9)
    t = float(np.clip(t, 0, 1))
    return _lerp(COLORS["dim"], COLORS["highlight"], t)

def _prob_color(p: float) -> ManimColor:
    """0 → dim, 1 → attention purple."""
    return _lerp(COLORS["dim"], COLORS["attention"], float(np.clip(p, 0, 1)))


# ─────────────────────────────────────────────
# Shared cell builder
# ─────────────────────────────────────────────
def _cell(
    value: float,
    color_fn,
    w: float = 0.52,
    h: float = 0.52,
    font_size: int = 13,
    show_val: bool = True,
    fmt: str = "{:+.2f}",
) -> VGroup:
    rect = Rectangle(
        width=w, height=h,
        fill_color=color_fn(value),
        fill_opacity=0.88,
        stroke_color=WHITE,
        stroke_width=0.7,
    )
    grp = VGroup(rect)
    if show_val:
        lbl = Text(fmt.format(value), font_size=font_size, color=WHITE)
        lbl.move_to(rect.get_center())
        grp.add(lbl)
    return grp


# ═══════════════════════════════════════════════════════════════════════════════
# Asset 1 — DotProductAnim
# ═══════════════════════════════════════════════════════════════════════════════

class DotProductAnim(VGroup):
    """
    Animates Q[i] · K[j] = score.

    Layout
    ──────
    Q vector  (horizontal, pink)
        ×  (element-wise)
    K vector  (horizontal, blue)
    ──────────────────────────────
    Products  (one cell per dim)
        ∑
    Score (single scalar box)

    Parameters
    ----------
    q_vec       : query vector (list of floats, length d_k)
    k_vec       : key   vector (list of floats, length d_k)
    q_label     : label string, default "Q[i]"
    k_label     : label string, default "K[j]"
    cell_size   : (w, h) of each cell
    font_size   : text size
    """

    def __init__(
        self,
        q_vec: Sequence[float],
        k_vec: Sequence[float],
        q_label: str = "Q[i]",
        k_label: str = "K[j]",
        cell_size: tuple[float, float] = (0.48, 0.48),
        font_size: int = 14,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(q_vec) == len(k_vec), "Q and K must have equal length"
        self.q_vec = list(q_vec)
        self.k_vec = list(k_vec)
        self.d_k = len(q_vec)
        self.q_label = q_label
        self.k_label = k_label
        self.cw, self.ch = cell_size
        self.fs = font_size
        self.products = [q * k for q, k in zip(self.q_vec, self.k_vec)]
        self.score = float(sum(self.products))

        self._build()
        self.add(
            self.q_row_group, self.times_sym,
            self.k_row_group, self.divider,
            self.prod_row_group, self.sum_sym,
            self.score_box,
        )

    def _make_row(self, values, color_hex, label_str, fmt="{:+.2f}"):
        cells = VGroup()
        for v in values:
            c = _cell(v, lambda x, ch=color_hex: _lerp(COLORS["dim"], ch,
                      float(np.clip(abs(x) / 1.5, 0, 1))),
                      self.cw, self.ch, self.fs, True, fmt)
            cells.add(c)
        cells.arrange(RIGHT, buff=0.04)
        lbl = Text(label_str, font_size=self.fs + 1,
                   color=ManimColor(color_hex))
        lbl.next_to(cells, LEFT, buff=0.2)
        return VGroup(lbl, cells), cells

    def _build(self):
        self.q_row_group, self.q_cells = self._make_row(
            self.q_vec, COLORS["query"], self.q_label)
        self.k_row_group, self.k_cells = self._make_row(
            self.k_vec, COLORS["key"], self.k_label)
        self.prod_row_group, self.prod_cells = self._make_row(
            self.products, COLORS["value"], "qᵢkᵢ", fmt="{:+.2f}")

        self.times_sym = MathTex(r"\times", font_size=28, color=WHITE)
        self.sum_sym   = MathTex(r"\sum", font_size=36, color=WHITE)

        score_rect = Rectangle(
            width=self.cw * 1.8, height=self.ch * 1.2,
            fill_color=_score_color(self.score),
            fill_opacity=0.9,
            stroke_color=_hex("highlight"),
            stroke_width=2,
        )
        score_lbl = Text(f"score\n{self.score:+.3f}", font_size=self.fs,
                         color=WHITE)
        score_lbl.move_to(score_rect.get_center())
        self.score_box = VGroup(score_rect, score_lbl)

        # Stack vertically
        self.q_row_group.move_to(ORIGIN)
        self.times_sym.next_to(self.q_row_group, DOWN, buff=0.08)
        self.k_row_group.next_to(self.times_sym, DOWN, buff=0.08)

        # Align k_row cells under q_row cells
        self.k_cells.align_to(self.q_cells, LEFT)

        # Divider line
        self.divider = Line(
            self.k_row_group.get_left() + LEFT * 0.1,
            self.k_row_group.get_right() + RIGHT * 0.1,
            color=GRAY, stroke_width=1,
        ).next_to(self.k_row_group, DOWN, buff=0.1)

        self.prod_row_group.next_to(self.divider, DOWN, buff=0.1)
        self.prod_cells.align_to(self.q_cells, LEFT)

        self.sum_sym.next_to(self.prod_row_group, DOWN, buff=0.1)
        self.score_box.next_to(self.sum_sym, DOWN, buff=0.1)

    # ── Animation API ─────────────────────────────────────────────────────────

    def build_anim(self, run_time: float = 0.5) -> Succession:
        steps = []

        # 1. Show Q and K rows
        steps.append(AnimationGroup(
            FadeIn(self.q_row_group, shift=RIGHT * 0.1),
            FadeIn(self.times_sym),
            FadeIn(self.k_row_group, shift=RIGHT * 0.1),
            FadeIn(self.divider),
            lag_ratio=0.2, run_time=run_time,
        ))

        # 2. Multiply element by element
        prod_anims = []
        for qc, kc, pc in zip(self.q_cells, self.k_cells, self.prod_cells):
            prod_anims.append(AnimationGroup(
                Indicate(qc, color=_hex("query"), scale_factor=1.15, run_time=run_time * 0.6),
                Indicate(kc, color=_hex("key"),   scale_factor=1.15, run_time=run_time * 0.6),
                FadeIn(pc, scale=0.7, run_time=run_time * 0.5),
                lag_ratio=0.3,
            ))
        steps.append(LaggedStart(*prod_anims, lag_ratio=0.25))

        # 3. Sum & score
        steps.append(AnimationGroup(
            Write(self.sum_sym, run_time=run_time * 0.6),
            FadeIn(self.score_box, scale=0.8, run_time=run_time),
            lag_ratio=0.4,
        ))

        # 4. Flash the score
        steps.append(Flash(
            self.score_box, color=_hex("highlight"),
            flash_radius=0.45, run_time=run_time * 0.7,
        ))

        return Succession(*steps)

    def highlight_score(self) -> Indicate:
        return Indicate(self.score_box, color=_hex("highlight"), scale_factor=1.1)

    def connect_to(self, other: VMobject) -> Arrow:
        return Arrow(
            self.score_box.get_bottom(),
            other.get_top(),
            buff=0.1,
            color=_hex("arrow"),
            stroke_width=2,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Asset 2 — ScoreMatrix
# ═══════════════════════════════════════════════════════════════════════════════

class ScoreMatrix(VGroup):
    """
    seq_len × seq_len grid of raw attention scores (QKᵀ).

    Fills row by row with animated color. Supports row/column highlight
    and individual cell tooltip.

    Parameters
    ----------
    scores      : 2-D array-like of shape (seq_len, seq_len)
    token_labels: list of token strings for axis labels
    cell_size   : (w, h)
    show_values : show float values inside cells
    """

    def __init__(
        self,
        scores: Sequence[Sequence[float]],
        token_labels: Sequence[str] | None = None,
        cell_size: tuple[float, float] = (0.50, 0.50),
        show_values: bool = True,
        font_size: int = 12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scores = [list(r) for r in scores]
        self.n = len(self.scores)
        assert all(len(r) == self.n for r in self.scores), "scores must be square"
        self.token_labels = token_labels or [f"t{i}" for i in range(self.n)]
        self.cw, self.ch = cell_size
        self.show_values = show_values
        self.fs = font_size

        lo = min(v for r in self.scores for v in r)
        hi = max(v for r in self.scores for v in r)
        self._lo, self._hi = lo, hi

        self._build()
        self.add(
            self.cell_grid,
            self.row_labels, self.col_labels,
            self.row_axis_title, self.col_axis_title,
        )

    def _score_col(self, v):
        t = (v - self._lo) / (self._hi - self._lo + 1e-9)
        return _lerp(COLORS["dim"], COLORS["highlight"], float(np.clip(t, 0, 1)))

    def _build(self):
        self.cell_mobs: list[list[VGroup]] = []
        grid = VGroup()
        for i, row in enumerate(self.scores):
            row_grp = VGroup()
            row_cells = []
            for j, val in enumerate(row):
                c = _cell(val, self._score_col, self.cw, self.ch,
                          self.fs, self.show_values, "{:+.1f}")
                row_grp.add(c)
                row_cells.append(c)
            row_grp.arrange(RIGHT, buff=0.03)
            grid.add(row_grp)
            self.cell_mobs.append(row_cells)
        grid.arrange(DOWN, buff=0.03)
        self.cell_grid = grid

        # Row labels (left) — "from" tokens
        self.row_labels = VGroup()
        for i, lbl in enumerate(self.token_labels):
            t = Text(f'"{lbl}"', font_size=self.fs + 1, color=_hex("query"))
            t.next_to(self.cell_mobs[i][0], LEFT, buff=0.18)
            self.row_labels.add(t)

        # Column labels (top) — "to" tokens
        self.col_labels = VGroup()
        for j, lbl in enumerate(self.token_labels):
            t = Text(f'"{lbl}"', font_size=self.fs + 1, color=_hex("key"))
            t.next_to(self.cell_mobs[0][j], UP, buff=0.18)
            self.col_labels.add(t)

        # Axis titles
        self.row_axis_title = Text("Query (from)", font_size=self.fs + 2,
                                   color=_hex("query"))
        self.row_axis_title.next_to(self.row_labels, LEFT, buff=0.15)
        self.row_axis_title.rotate(90 * DEGREES)

        self.col_axis_title = Text("Key (to)", font_size=self.fs + 2,
                                   color=_hex("key"))
        self.col_axis_title.next_to(self.col_labels, UP, buff=0.15)

    # ── Animation API ─────────────────────────────────────────────────────────

    def build_anim(self, run_time_per_row: float = 0.4) -> Succession:
        """Reveal row labels → col labels → cells row by row."""
        steps = []

        steps.append(AnimationGroup(
            FadeIn(self.row_axis_title), FadeIn(self.col_axis_title),
            FadeIn(self.row_labels), FadeIn(self.col_labels),
            lag_ratio=0.2, run_time=run_time_per_row,
        ))

        for i, row_cells in enumerate(self.cell_mobs):
            row_anims = [FadeIn(c, scale=0.6) for c in row_cells]
            steps.append(LaggedStart(*row_anims, lag_ratio=0.12,
                                     run_time=run_time_per_row))

        return Succession(*steps)

    def highlight_cell(self, i: int, j: int) -> AnimationGroup:
        """Pulse cell (i, j) and return its score."""
        c = self.cell_mobs[i][j]
        score = self.scores[i][j]
        tooltip = Text(
            f'"{self.token_labels[i]}" → "{self.token_labels[j]}"\n'
            f'score = {score:+.2f}',
            font_size=self.fs + 1, color=WHITE,
        )
        tooltip.next_to(c, RIGHT, buff=0.2)
        return AnimationGroup(
            Indicate(c, color=_hex("highlight"), scale_factor=1.3),
            FadeIn(tooltip, shift=RIGHT * 0.1),
            lag_ratio=0.1,
        )

    def highlight_row(self, i: int) -> AnimationGroup:
        return AnimationGroup(*[
            Indicate(c, color=_hex("query"), scale_factor=1.1)
            for c in self.cell_mobs[i]
        ], lag_ratio=0.05)

    def highlight_col(self, j: int) -> AnimationGroup:
        return AnimationGroup(*[
            Indicate(self.cell_mobs[i][j], color=_hex("key"), scale_factor=1.1)
            for i in range(self.n)
        ], lag_ratio=0.05)

    def get_cell(self, i: int, j: int) -> VGroup:
        return self.cell_mobs[i][j]


# ═══════════════════════════════════════════════════════════════════════════════
# Asset 3 — ScalingAnim
# ═══════════════════════════════════════════════════════════════════════════════

class ScalingAnim(VGroup):
    """
    Divides raw score matrix by √dₖ.

    Shows:  raw scores  ÷  √dₖ  =  scaled scores
    with a before/after comparison and the formula.

    Parameters
    ----------
    scores  : raw score matrix (seq_len × seq_len)
    d_k     : key dimension (used to compute √dₖ)
    """

    def __init__(
        self,
        scores: Sequence[Sequence[float]],
        d_k: int = 64,
        token_labels: Sequence[str] | None = None,
        cell_size: tuple[float, float] = (0.46, 0.46),
        font_size: int = 13,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw = [list(r) for r in scores]
        self.n = len(self.raw)
        self.d_k = d_k
        self.scale_factor = math.sqrt(d_k)
        self.scaled = [[v / self.scale_factor for v in row] for row in self.raw]
        self.token_labels = token_labels or [f"t{i}" for i in range(self.n)]
        self.cw, self.ch = cell_size
        self.fs = font_size

        self._build()
        self.add(
            self.raw_group, self.div_group,
            self.scaled_group, self.formula,
        )

    def _matrix_mob(self, data, title_str, title_color_key):
        lo = min(v for r in data for v in r)
        hi = max(v for r in data for v in r)

        def col(v):
            t = (v - lo) / (hi - lo + 1e-9)
            return _lerp(COLORS["dim"], COLORS["highlight"], float(np.clip(t, 0, 1)))

        grid = VGroup()
        self.cell_refs = []
        for row in data:
            rg = VGroup()
            for v in row:
                rg.add(_cell(v, col, self.cw, self.ch, self.fs, True, "{:.2f}"))
            rg.arrange(RIGHT, buff=0.03)
            grid.add(rg)
        grid.arrange(DOWN, buff=0.03)

        title = Text(title_str, font_size=self.fs + 2, color=_hex(title_color_key))
        title.next_to(grid, UP, buff=0.12)
        return VGroup(title, grid), grid

    def _build(self):
        self.raw_group, self.raw_grid = self._matrix_mob(
            self.raw, "Raw Scores (QKᵀ)", "query")

        div_sym = MathTex(
            r"\div\;\sqrt{d_k}",
            font_size=30, color=WHITE,
        )
        dk_val = Text(f"√{self.d_k} = {self.scale_factor:.1f}",
                      font_size=self.fs, color=GRAY)
        self.div_group = VGroup(div_sym, dk_val)
        self.div_group.arrange(DOWN, buff=0.1)

        self.scaled_group, self.scaled_grid = self._matrix_mob(
            self.scaled, "Scaled Scores", "key")

        # Arrange horizontally
        self.raw_group.move_to(ORIGIN)
        self.div_group.next_to(self.raw_group, RIGHT, buff=0.4)
        self.scaled_group.next_to(self.div_group, RIGHT, buff=0.4)

        # Formula below
        self.formula = MathTex(
            r"\text{score}_{ij} = \frac{Q_i \cdot K_j}{\sqrt{d_k}}",
            font_size=22, color=GRAY,
        )
        self.formula.next_to(self.scaled_group, DOWN, buff=0.3)

    # ── Animation API ─────────────────────────────────────────────────────────

    def build_anim(self, run_time: float = 0.6) -> Succession:
        steps = []

        # 1. Show raw matrix
        steps.append(FadeIn(self.raw_group, run_time=run_time))

        # 2. Show ÷√dₖ
        steps.append(FadeIn(self.div_group, shift=RIGHT * 0.1, run_time=run_time * 0.7))

        # 3. Reveal scaled matrix cell by cell (Transform effect)
        anims = []
        for rg, sg in zip(self.raw_grid, self.scaled_grid):
            for rc, sc in zip(rg, sg):
                anims.append(FadeIn(sc, scale=0.7))
        steps.append(LaggedStart(*anims, lag_ratio=0.06, run_time=run_time * 2))

        # 4. Formula
        steps.append(Write(self.formula, run_time=run_time))

        return Succession(*steps)

    def show_before_after(self) -> AnimationGroup:
        """Indicate both matrices simultaneously for comparison."""
        return AnimationGroup(
            Indicate(self.raw_grid,    color=_hex("query"), scale_factor=1.05),
            Indicate(self.scaled_grid, color=_hex("key"),   scale_factor=1.05),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Asset 4 — CausalMask
# ═══════════════════════════════════════════════════════════════════════════════

class CausalMask(VGroup):
    """
    Overlays a causal (autoregressive) mask on a score matrix.

    Upper triangle becomes -∞ (dark blocked cells).
    Lower triangle stays bright (allowed positions).
    A diagonal highlight and "Future blocked" label animate in.

    Parameters
    ----------
    scores          : raw or scaled score matrix (will be masked)
    token_labels    : token strings
    show_minus_inf  : whether to write "-∞" in blocked cells
    """

    def __init__(
        self,
        scores: Sequence[Sequence[float]],
        token_labels: Sequence[str] | None = None,
        cell_size: tuple[float, float] = (0.50, 0.50),
        font_size: int = 13,
        show_minus_inf: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scores = [list(r) for r in scores]
        self.n = len(self.scores)
        self.token_labels = token_labels or [f"t{i}" for i in range(self.n)]
        self.cw, self.ch = cell_size
        self.fs = font_size
        self.show_minus_inf = show_minus_inf

        self._build()
        self.add(
            self.base_grid,
            self.row_labels, self.col_labels,
            self.mask_overlays,
            self.diagonal_line,
            self.blocked_label, self.allowed_label,
        )

    def _build(self):
        lo = min(v for r in self.scores for v in r)
        hi = max(v for r in self.scores for v in r)

        def col(v):
            t = (v - lo) / (hi - lo + 1e-9)
            return _lerp(COLORS["dim"], COLORS["highlight"], float(np.clip(t, 0, 1)))

        # Base grid (all visible initially)
        self.cell_mobs: list[list[VGroup]] = []
        grid = VGroup()
        for i, row in enumerate(self.scores):
            rg = VGroup()
            row_cells = []
            for j, val in enumerate(row):
                c = _cell(val, col, self.cw, self.ch, self.fs, True, "{:.1f}")
                rg.add(c)
                row_cells.append(c)
            rg.arrange(RIGHT, buff=0.03)
            grid.add(rg)
            self.cell_mobs.append(row_cells)
        grid.arrange(DOWN, buff=0.03)
        self.base_grid = grid

        # Mask overlays (upper triangle, j > i)
        self.mask_overlays = VGroup()
        self.mask_cells: list[VGroup] = []
        for i in range(self.n):
            for j in range(self.n):
                if j > i:
                    base_cell = self.cell_mobs[i][j]
                    overlay = Rectangle(
                        width=self.cw, height=self.ch,
                        fill_color=ManimColor(COLORS.get("mask", "#1A1A2E")),
                        fill_opacity=0.95,
                        stroke_color=ManimColor("#333355"),
                        stroke_width=0.6,
                    )
                    overlay.move_to(base_cell.get_center())
                    if self.show_minus_inf:
                        inf_lbl = MathTex(r"-\infty", font_size=self.fs - 1,
                                          color=ManimColor("#555577"))
                        inf_lbl.move_to(overlay.get_center())
                        mask_mob = VGroup(overlay, inf_lbl)
                    else:
                        mask_mob = VGroup(overlay)
                    self.mask_overlays.add(mask_mob)
                    self.mask_cells.append(mask_mob)

        # Diagonal line (visual separator)
        tl = self.cell_mobs[0][0].get_corner([-1, 1, 0])   # top-left of (0,0)
        br = self.cell_mobs[self.n-1][self.n-1].get_corner([1, -1, 0])
        self.diagonal_line = Line(tl, br, color=_hex("highlight"),
                                  stroke_width=2)

        # Labels
        self.blocked_label = Text("← Future (blocked)", font_size=self.fs + 1,
                                  color=ManimColor("#555577"))
        self.blocked_label.next_to(self.cell_mobs[0][-1], UP + RIGHT, buff=0.1)

        self.allowed_label = Text("Past + present ✓", font_size=self.fs + 1,
                                  color=_hex("token"))
        self.allowed_label.next_to(self.cell_mobs[-1][0], DOWN + LEFT, buff=0.1)

        # Row/col labels
        self.row_labels = VGroup()
        self.col_labels = VGroup()
        for i, lbl in enumerate(self.token_labels):
            rt = Text(f'"{lbl}"', font_size=self.fs, color=_hex("query"))
            rt.next_to(self.cell_mobs[i][0], LEFT, buff=0.18)
            self.row_labels.add(rt)

            ct = Text(f'"{lbl}"', font_size=self.fs, color=_hex("key"))
            ct.next_to(self.cell_mobs[0][i], UP, buff=0.18)
            self.col_labels.add(ct)

        # Initially hide mask overlays and labels
        for m in self.mask_overlays:
            m.set_opacity(0)
        self.diagonal_line.set_opacity(0)
        self.blocked_label.set_opacity(0)
        self.allowed_label.set_opacity(0)

    # ── Animation API ─────────────────────────────────────────────────────────

    def build_anim(self, run_time: float = 0.5) -> Succession:
        """
        1. Show base grid + labels
        2. Draw diagonal
        3. Sweep mask over upper triangle (anti-diagonal waves)
        4. Reveal labels
        """
        steps = []

        # 1. Base grid
        steps.append(AnimationGroup(
            FadeIn(self.base_grid),
            FadeIn(self.row_labels),
            FadeIn(self.col_labels),
            lag_ratio=0.2, run_time=run_time,
        ))

        # 2. Diagonal line
        steps.append(Create(self.diagonal_line, run_time=run_time * 0.6))

        # 3. Mask cells — sweep by anti-diagonal
        # Group by i+j (anti-diagonal index)
        from collections import defaultdict
        diag_groups: dict[int, list] = defaultdict(list)
        idx = 0
        for i in range(self.n):
            for j in range(self.n):
                if j > i:
                    diag_groups[i + j].append(self.mask_cells[idx])
                    idx += 1

        sweep_anims = []
        for d in sorted(diag_groups):
            group_anim = AnimationGroup(*[
                m.animate.set_opacity(1) for m in diag_groups[d]
            ], run_time=run_time * 0.4)
            sweep_anims.append(group_anim)
        steps.append(LaggedStart(*sweep_anims, lag_ratio=0.3))

        # 4. Labels
        steps.append(AnimationGroup(
            self.blocked_label.animate.set_opacity(1),
            self.allowed_label.animate.set_opacity(1),
            run_time=run_time * 0.6,
        ))

        return Succession(*steps)

    def unmask_cell(self, i: int, j: int) -> FadeOut:
        """Reveal a masked cell (useful for debugging / explanation)."""
        if j <= i:
            raise ValueError(f"Cell ({i},{j}) is not masked (j must be > i)")
        # Find the matching mask cell
        idx = 0
        for ii in range(self.n):
            for jj in range(self.n):
                if jj > ii:
                    if ii == i and jj == j:
                        return FadeOut(self.mask_cells[idx])
                    idx += 1

    def highlight_diagonal(self) -> Indicate:
        return Indicate(self.diagonal_line, color=_hex("highlight"), scale_factor=1.1)


# ═══════════════════════════════════════════════════════════════════════════════
# Asset 5 — SoftmaxAnim
# ═══════════════════════════════════════════════════════════════════════════════

class SoftmaxAnim(VGroup):
    """
    Animates one row of scores → softmax probabilities.

    Shows:
      • Raw score bar chart
      • Softmax formula
      • Probability bar chart
      • "Sums to 1.0" indicator
      • Optional temperature slider effect

    Parameters
    ----------
    scores          : 1-D list of raw (scaled) scores for one query token
    token_labels    : key token labels (columns)
    temperature     : softmax temperature (default 1.0)
    show_formula    : whether to show the softmax equation
    """

    def __init__(
        self,
        scores: Sequence[float],
        token_labels: Sequence[str] | None = None,
        temperature: float = 1.0,
        show_formula: bool = True,
        bar_max_height: float = 1.6,
        font_size: int = 14,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_scores = list(scores)
        self.n = len(scores)
        self.token_labels = token_labels or [f"t{i}" for i in range(self.n)]
        self.temperature = temperature
        self.show_formula = show_formula
        self.bar_h = bar_max_height
        self.fs = font_size

        self.probs = self._softmax(self.raw_scores, temperature)
        self._build()
        self.add(
            self.raw_bar_group,
            self.arrow_group,
            self.prob_bar_group,
            self.formula_mob,
            self.sum_label,
        )

    @staticmethod
    def _softmax(scores: list[float], T: float = 1.0) -> list[float]:
        s = np.array(scores) / T
        s = s - s.max()
        e = np.exp(s)
        return (e / e.sum()).tolist()

    def _bar_chart(self, values, color_fn, label_prefix="") -> VGroup:
        group = VGroup()
        max_v = max(abs(v) for v in values) if values else 1.0
        for val, lbl in zip(values, self.token_labels):
            h = max(abs(val) / max_v * self.bar_h, 0.04)
            bar = Rectangle(
                width=0.48, height=h,
                fill_color=color_fn(val),
                fill_opacity=0.88,
                stroke_color=WHITE, stroke_width=0.7,
            )
            val_lbl = Text(f"{val:.2f}", font_size=self.fs - 2, color=WHITE)
            val_lbl.next_to(bar, UP, buff=0.05)
            tok_lbl = Text(f'"{lbl}"', font_size=self.fs - 3,
                           color=_hex("key"))
            tok_lbl.next_to(bar, DOWN, buff=0.08)
            col = VGroup(bar, val_lbl, tok_lbl)
            group.add(col)
        group.arrange(RIGHT, buff=0.18, aligned_edge=DOWN)
        return group

    def _build(self):
        # Raw score bars
        def raw_col(v):
            t = (v - min(self.raw_scores)) / (max(self.raw_scores) - min(self.raw_scores) + 1e-9)
            return _lerp(COLORS["dim"], COLORS["query"], float(np.clip(t, 0, 1)))

        raw_bars = self._bar_chart(self.raw_scores, raw_col)
        raw_title = Text("Raw Scores", font_size=self.fs + 1, color=_hex("query"))
        raw_title.next_to(raw_bars, UP, buff=0.18)
        self.raw_bar_group = VGroup(raw_title, raw_bars)
        self.raw_bars_mob = raw_bars

        # Arrow + formula
        arr = Arrow(LEFT * 0.1, RIGHT * 0.1, color=WHITE, stroke_width=2)
        sm_formula = MathTex(
            r"\text{softmax}\!\left(\frac{s}{\sqrt{d_k}}\right)",
            font_size=18, color=_hex("attention"),
        ) if self.show_formula else Text("softmax", font_size=self.fs, color=_hex("attention"))
        self.arrow_group = VGroup(arr, sm_formula)
        self.arrow_group.arrange(DOWN, buff=0.08)

        # Probability bars
        def prob_col(v):
            return _lerp(COLORS["dim"], COLORS["attention"], float(np.clip(v, 0, 1)))

        prob_bars = self._bar_chart(self.probs, prob_col)
        prob_title = Text("Attention Weights", font_size=self.fs + 1, color=_hex("attention"))
        prob_title.next_to(prob_bars, UP, buff=0.18)
        self.prob_bar_group = VGroup(prob_title, prob_bars)
        self.prob_bars_mob = prob_bars

        # Arrange horizontally
        self.raw_bar_group.move_to(ORIGIN)
        self.arrow_group.next_to(self.raw_bar_group, RIGHT, buff=0.35)
        self.prob_bar_group.next_to(self.arrow_group, RIGHT, buff=0.35)

        # Formula (below prob bars)
        self.formula_mob = MathTex(
            r"\sum_j w_j = 1.0",
            font_size=20, color=GRAY,
        ) if self.show_formula else VGroup()
        self.formula_mob.next_to(self.prob_bar_group, DOWN, buff=0.2)

        # Sum indicator
        total = sum(self.probs)
        self.sum_label = Text(
            f"Σ = {total:.4f}  ✓",
            font_size=self.fs, color=_hex("token"),
        )
        self.sum_label.next_to(self.formula_mob, DOWN, buff=0.1)

    # ── Animation API ─────────────────────────────────────────────────────────

    def build_anim(self, run_time: float = 0.6) -> Succession:
        steps = []

        # 1. Raw bars grow up
        raw_bar_anims = [
            bar[0].animate.set_fill(opacity=0.88)
            for bar in self.raw_bars_mob
        ]
        steps.append(LaggedStart(
            *[FadeIn(col, shift=UP * 0.15) for col in self.raw_bars_mob],
            lag_ratio=0.15, run_time=run_time,
        ))
        steps.append(FadeIn(self.raw_bar_group[0], run_time=run_time * 0.4))

        # 2. Arrow + formula
        steps.append(AnimationGroup(
            GrowArrow(self.arrow_group[0]),
            FadeIn(self.arrow_group[1] if len(self.arrow_group) > 1 else VGroup()),
            run_time=run_time * 0.7,
        ))

        # 3. Prob bars appear
        steps.append(LaggedStart(
            *[FadeIn(col, shift=UP * 0.15) for col in self.prob_bars_mob],
            lag_ratio=0.15, run_time=run_time,
        ))
        steps.append(FadeIn(self.prob_bar_group[0], run_time=run_time * 0.4))

        # 4. Formula + sum
        steps.append(AnimationGroup(
            Write(self.formula_mob),
            FadeIn(self.sum_label, shift=UP * 0.05),
            lag_ratio=0.3, run_time=run_time,
        ))

        # 5. Flash top-weight bar
        top_idx = int(np.argmax(self.probs))
        top_bar_col = self.prob_bars_mob[top_idx]
        steps.append(Flash(top_bar_col, color=_hex("highlight"),
                           flash_radius=0.35, run_time=run_time * 0.6))

        return Succession(*steps)

    def show_temperature_effect(
        self, new_temp: float, scene: Scene, run_time: float = 0.8
    ):
        """
        In-place morphs bars to reflect a new temperature.
        Call from scene.construct() — not a standalone animation object.
        """
        new_probs = self._softmax(self.raw_scores, new_temp)
        max_old = max(self.probs)
        max_new = max(new_probs)
        temp_label = Text(
            f"T = {new_temp:.1f}", font_size=self.fs + 2,
            color=_hex("highlight"),
        )
        temp_label.next_to(self.prob_bar_group, UP, buff=0.15)
        scene.play(FadeIn(temp_label, run_time=run_time * 0.4))

        anims = []
        for i, (col, new_p) in enumerate(zip(self.prob_bars_mob, new_probs)):
            new_h = max(new_p / max_new * self.bar_h, 0.04)
            bar_rect = col[0]
            anims.append(bar_rect.animate.stretch_to_fit_height(new_h)
                         .align_to(col, DOWN))
        scene.play(AnimationGroup(*anims, lag_ratio=0.05, run_time=run_time))


# ═══════════════════════════════════════════════════════════════════════════════
# Asset 6 — AttentionHeatmap
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionHeatmap(VGroup):
    """
    Final attention weight matrix displayed as a color heatmap.

    High attention = bright purple/yellow.
    Token labels on both axes.
    Animates row by row with a "Token X attends to Token Y" caption.

    Parameters
    ----------
    weights         : 2-D softmax weight matrix (seq_len × seq_len)
    token_labels    : token strings
    cell_size       : (w, h)
    show_values     : render probability values inside cells
    """

    def __init__(
        self,
        weights: Sequence[Sequence[float]],
        token_labels: Sequence[str] | None = None,
        cell_size: tuple[float, float] = (0.55, 0.55),
        show_values: bool = True,
        font_size: int = 12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weights = [list(r) for r in weights]
        self.n = len(self.weights)
        self.token_labels = token_labels or [f"t{i}" for i in range(self.n)]
        self.cw, self.ch = cell_size
        self.show_values = show_values
        self.fs = font_size

        self._build()
        self.add(
            self.cell_grid,
            self.row_labels, self.col_labels,
            self.colorbar,
            self.title_mob,
        )

    def _build(self):
        self.cell_mobs: list[list[VGroup]] = []
        grid = VGroup()
        for i, row in enumerate(self.weights):
            rg = VGroup()
            row_cells = []
            for j, w in enumerate(row):
                rect = Rectangle(
                    width=self.cw, height=self.ch,
                    fill_color=_prob_color(w),
                    fill_opacity=0.92,
                    stroke_color=WHITE, stroke_width=0.5,
                )
                cell = VGroup(rect)
                if self.show_values:
                    lbl = Text(f"{w:.2f}", font_size=self.fs - 1, color=WHITE)
                    lbl.move_to(rect.get_center())
                    cell.add(lbl)
                rg.add(cell)
                row_cells.append(cell)
            rg.arrange(RIGHT, buff=0.02)
            grid.add(rg)
            self.cell_mobs.append(row_cells)
        grid.arrange(DOWN, buff=0.02)
        self.cell_grid = grid

        # Labels
        self.row_labels = VGroup()
        self.col_labels = VGroup()
        for i, lbl in enumerate(self.token_labels):
            rt = Text(f'"{lbl}"', font_size=self.fs + 1, color=_hex("query"))
            rt.next_to(self.cell_mobs[i][0], LEFT, buff=0.18)
            self.row_labels.add(rt)

            ct = Text(f'"{lbl}"', font_size=self.fs + 1, color=_hex("key"))
            ct.next_to(self.cell_mobs[0][i], UP, buff=0.18)
            self.col_labels.add(ct)

        # Color bar (dim→bright legend)
        bar = VGroup()
        n_steps = 12
        for k in range(n_steps):
            t = k / (n_steps - 1)
            seg = Rectangle(
                width=0.18, height=self.ch * 0.55,
                fill_color=_prob_color(t), fill_opacity=0.92,
                stroke_width=0,
            )
            bar.add(seg)
        bar.arrange(RIGHT, buff=0)
        lo_lbl = Text("0.0", font_size=self.fs - 2, color=GRAY)
        hi_lbl = Text("1.0", font_size=self.fs - 2, color=GRAY)
        lo_lbl.next_to(bar, LEFT, buff=0.08)
        hi_lbl.next_to(bar, RIGHT, buff=0.08)
        self.colorbar = VGroup(lo_lbl, bar, hi_lbl)
        self.colorbar.next_to(grid, DOWN, buff=0.3)

        # Title
        self.title_mob = Text(
            "Attention Weights", font_size=self.fs + 4,
            color=_hex("attention"),
        )
        self.title_mob.next_to(self.col_labels, UP, buff=0.2)

        # Caption (initially empty)
        self._caption = Text("", font_size=self.fs + 1, color=WHITE)
        self._caption.next_to(self.colorbar, DOWN, buff=0.12)
        self.add(self._caption)

    # ── Animation API ─────────────────────────────────────────────────────────

    def build_anim(self, run_time_per_row: float = 0.4) -> Succession:
        """Reveal title → labels → heatmap cells row by row."""
        steps = []

        steps.append(AnimationGroup(
            Write(self.title_mob),
            FadeIn(self.row_labels),
            FadeIn(self.col_labels),
            FadeIn(self.colorbar),
            lag_ratio=0.2, run_time=run_time_per_row,
        ))

        for i, row_cells in enumerate(self.cell_mobs):
            steps.append(LaggedStart(
                *[FadeIn(c, scale=0.5) for c in row_cells],
                lag_ratio=0.1, run_time=run_time_per_row,
            ))

        return Succession(*steps)

    def annotate_cell(self, i: int, j: int, scene: Scene, run_time: float = 0.5):
        """
        Highlight cell (i,j) and show caption.
        Call from scene.construct().
        """
        cell = self.cell_mobs[i][j]
        w = self.weights[i][j]
        caption_str = (
            f'"{self.token_labels[i]}" attends to '
            f'"{self.token_labels[j]}"  →  {w:.2f}'
        )
        new_caption = Text(caption_str, font_size=self.fs + 1, color=WHITE)
        new_caption.move_to(self._caption.get_center())

        box = SurroundingRectangle(cell, color=_hex("highlight"), buff=0.04, stroke_width=2)
        scene.play(
            Create(box),
            Transform(self._caption, new_caption),
            Indicate(cell, color=_hex("highlight"), scale_factor=1.2),
            run_time=run_time,
        )
        scene.play(FadeOut(box), run_time=run_time * 0.4)

    def highlight_row(self, i: int) -> AnimationGroup:
        return AnimationGroup(*[
            Indicate(c, color=_hex("query"), scale_factor=1.1)
            for c in self.cell_mobs[i]
        ], lag_ratio=0.04)

    def highlight_col(self, j: int) -> AnimationGroup:
        return AnimationGroup(*[
            Indicate(self.cell_mobs[i][j], color=_hex("key"), scale_factor=1.1)
            for i in range(self.n)
        ], lag_ratio=0.04)

    def connect_from(self, source: VMobject) -> CurvedArrow:
        return CurvedArrow(
            source.get_right() + RIGHT * 0.05,
            self.get_left() + LEFT * 0.05,
            angle=-0.25, color=_hex("arrow"), stroke_width=2,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Demo scene — manim -pql attention_score.py DemoScene
# ═══════════════════════════════════════════════════════════════════════════════

class DemoScene(Scene):
    """Runs all 6 assets back-to-back."""

    def construct(self):
        tokens = ["the", "cat", "sat", "on"]
        n = len(tokens)
        rng = np.random.default_rng(42)

        # ── 1. DotProductAnim ─────────────────────────────────────────────────
        q = rng.uniform(-1, 1, 6).tolist()
        k = rng.uniform(-1, 1, 6).tolist()
        dp = DotProductAnim(q, k, q_label="Q[cat]", k_label="K[the]", font_size=12)
        dp.scale(0.75).move_to(ORIGIN)
        self.play(dp.build_anim(run_time=0.4))
        self.wait(0.4)
        self.play(FadeOut(dp))

        # ── 2. ScoreMatrix ────────────────────────────────────────────────────
        raw_scores = rng.uniform(-2, 2, (n, n)).tolist()
        sm = ScoreMatrix(raw_scores, token_labels=tokens, font_size=12)
        sm.scale(0.75).move_to(ORIGIN)
        self.play(sm.build_anim(run_time_per_row=0.35))
        self.play(sm.highlight_cell(1, 0))
        self.wait(0.3)
        self.play(FadeOut(sm))

        # ── 3. ScalingAnim ────────────────────────────────────────────────────
        sa = ScalingAnim(raw_scores, d_k=64, token_labels=tokens, font_size=12)
        sa.scale(0.7).move_to(ORIGIN)
        self.play(sa.build_anim(run_time=0.45))
        self.wait(0.3)
        self.play(FadeOut(sa))

        # ── 4. CausalMask ─────────────────────────────────────────────────────
        scaled = [[v / 8.0 for v in row] for row in raw_scores]
        cm = CausalMask(scaled, token_labels=tokens, font_size=12)
        cm.scale(0.78).move_to(ORIGIN)
        self.play(cm.build_anim(run_time=0.4))
        self.wait(0.4)
        self.play(FadeOut(cm))

        # ── 5. SoftmaxAnim ────────────────────────────────────────────────────
        # Use lower-triangle row 2 of scaled scores
        row_scores = [scaled[2][j] if j <= 2 else -1e9 for j in range(n)]
        sfx = SoftmaxAnim(row_scores, token_labels=tokens, font_size=13)
        sfx.scale(0.72).move_to(ORIGIN)
        self.play(sfx.build_anim(run_time=0.5))
        self.wait(0.4)
        self.play(FadeOut(sfx))

        # ── 6. AttentionHeatmap ───────────────────────────────────────────────
        # Build proper softmax weights for the full matrix
        def sm_row(row):
            s = np.array(row)
            s -= s.max()
            e = np.exp(s)
            return (e / e.sum()).tolist()

        weights = [sm_row(row) for row in scaled]
        hm = AttentionHeatmap(weights, token_labels=tokens, font_size=12)
        hm.scale(0.80).move_to(ORIGIN)
        self.play(hm.build_anim(run_time_per_row=0.35))
        hm.annotate_cell(1, 0, self, run_time=0.5)
        self.play(hm.highlight_row(2))
        self.wait(0.5)