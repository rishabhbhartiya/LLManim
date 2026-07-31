"""
embeddings/embedding_lookup.py
================================
Manim assets for visualizing the embedding lookup process in Transformers.

Assets
------
EmbeddingMatrix       — Large vocab × d_model grid with row-extraction animation
TokenToVectorAnim     — Full pipeline: token box → matrix lookup → vector output
VectorDisplay         — Horizontal bar showing d_model dimensions with value colors
VectorSpaceViz        — 2D scatter plot of token embeddings in semantic space
EmbeddingAddition     — Element-wise vector addition animation

Usage
-----
    from embeddings.embedding_lookup import EmbeddingMatrix, TokenToVectorAnim

    class MyScene(Scene):
        def construct(self):
            mat = EmbeddingMatrix(vocab_size=16, d_model=8)
            self.add(mat)
            self.play(mat.lookup_row(3))
"""

from __future__ import annotations

from manim import (
    VGroup, Rectangle, Text, MathTex, Tex,
    Arrow, CurvedArrow, Line, Dot, Axes,
    Scene, Animation, AnimationGroup, Succession,
    FadeIn, FadeOut, Create, Write,
    Transform, ReplacementTransform,
    Indicate, Flash, Circumscribe,
    LaggedStart, LaggedStartMap,
    GrowArrow, MoveToTarget,
    LEFT, RIGHT, UP, DOWN, ORIGIN,
    WHITE, BLACK, GRAY, DARK_GRAY, LIGHT_GRAY,
    RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, PINK,
    interpolate_color, color_to_rgb,
    ValueTracker, always_redraw,
    rate_functions,
    config,
)
from manim import Brace, DecimalNumber

import numpy as _np  # internal alias — Manim re-exports np, so we use _np directly

# ---------------------------------------------------------------------------
# Import shared styles (graceful fallback so file can be read in isolation)
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
    CELL_SIZE    = 0.45
    SMALL_CELL   = 0.28
    LABEL_SCALE  = 0.35
    DEFAULT_RUN_TIME = 1.0
    FAST_RUN_TIME    = 0.4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _value_to_color(value: float, v_min: float = -1.0, v_max: float = 1.0):
    """Map a scalar in [v_min, v_max] → red (negative) / white (zero) / blue (positive)."""
    t = (value - v_min) / (v_max - v_min + 1e-9)  # 0 → 1
    if t < 0.5:
        return interpolate_color(RED, WHITE, t * 2)
    else:
        return interpolate_color(WHITE, BLUE, (t - 0.5) * 2)


def _dim_label(text: str, scale: float = LABEL_SCALE) -> Text:
    return Text(text, font="Monospace").scale(scale).set_color(LIGHT_GRAY)


# ===========================================================================
# 1.  EmbeddingMatrix
# ===========================================================================

class EmbeddingMatrix(VGroup):
    """
    Visual: a (vocab_size × d_model) grid where most rows are dimmed and one
    row can be highlighted / extracted.

    Parameters
    ----------
    vocab_size : int   — number of rows shown (can be larger than displayed; use ellipsis)
    d_model    : int   — number of columns (dimensions)
    show_values: bool  — show floating-point numbers inside cells
    cell_size  : float — side length of each cell square
    max_rows   : int   — cap displayed rows (adds "…" if vocab_size > max_rows)
    """

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 8,
        show_values: bool = False,
        cell_size: float = SMALL_CELL,
        max_rows: int = 12,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.cell_size   = cell_size
        self.show_values = show_values

        # Fake weight values (deterministic for reproducibility)
        rng = _np.random.default_rng(42)
        self._values = rng.uniform(-1, 1, (vocab_size, d_model))

        # ── build grid ──────────────────────────────────────────────────────
        self._rows: list[VGroup] = []
        self._cells: list[list[Rectangle]] = []

        n_displayed = min(vocab_size, max_rows)
        has_ellipsis = vocab_size > max_rows

        for r in range(n_displayed):
            row_group = VGroup()
            cell_row  = []
            for c in range(d_model):
                val  = self._values[r, c]
                fill = _value_to_color(val)
                rect = Rectangle(
                    width=cell_size, height=cell_size,
                    fill_color=fill, fill_opacity=0.25,
                    stroke_color=COLORS["dim"], stroke_width=0.8,
                )
                rect.move_to(
                    _np.array([c * cell_size, -r * cell_size, 0])
                )
                if show_values:
                    num = Text(f"{val:.1f}", font="Monospace").scale(0.18)
                    num.move_to(rect.get_center())
                    row_group.add(num)

                row_group.add(rect)
                cell_row.append(rect)
            self._rows.append(row_group)
            self._cells.append(cell_row)
            self.add(row_group)

        if has_ellipsis:
            dots = Text("⋮", font="Monospace").scale(0.5).set_color(GRAY)
            dots.next_to(self._rows[-1], DOWN * 0.4)
            self.add(dots)

        # ── axis braces / labels ────────────────────────────────────────────
        brace_vocab  = Brace(self, LEFT,  buff=0.1)
        label_vocab  = brace_vocab.get_text(f"vocab\n({vocab_size})")
        label_vocab.scale(LABEL_SCALE).set_color(LIGHT_GRAY)

        brace_d      = Brace(self, UP, buff=0.1)
        label_d      = brace_d.get_text(f"d_model ({d_model})")
        label_d.scale(LABEL_SCALE).set_color(LIGHT_GRAY)

        self._brace_group = VGroup(brace_vocab, label_vocab, brace_d, label_d)
        self.add(self._brace_group)

        # Title label
        title = Text("Embedding Matrix  W_E", font="Monospace") \
                    .scale(LABEL_SCALE + 0.05) \
                    .set_color(COLORS["embedding"]) \
                    .next_to(self, UP, buff=0.35)
        self.add(title)

    # ── public animation methods ─────────────────────────────────────────────

    def dim_all_rows(self, opacity: float = 0.15) -> Animation:
        """Fade all rows to low opacity (call before a lookup)."""
        anims = []
        for row in self._rows:
            anims.append(row.animate.set_opacity(opacity))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def highlight_row(self, row_idx: int, run_time: float = DEFAULT_RUN_TIME) -> AnimationGroup:
        """
        Brighten row `row_idx`, flash its cells, and show a token-id label
        on the left margin.
        """
        if row_idx >= len(self._rows):
            raise IndexError(f"row_idx {row_idx} out of range ({len(self._rows)} rows shown)")

        row = self._rows[row_idx]
        cells = self._cells[row_idx]

        anims = [
            row.animate.set_opacity(1.0),
            Flash(row, color=COLORS["highlight"], line_length=0.15, run_time=run_time),
        ]
        for cell in cells:
            anims.append(
                cell.animate(run_time=run_time * 0.6)
                    .set_stroke(color=COLORS["highlight"], width=2.5)
                    .set_fill(opacity=0.75)
            )
        return AnimationGroup(*anims, lag_ratio=0.0, run_time=run_time)

    def extract_row(
        self,
        row_idx: int,
        target_position=None,
        run_time: float = DEFAULT_RUN_TIME,
    ) -> tuple[VGroup, Animation]:
        """
        Slides a copy of the highlighted row out to `target_position`.

        Returns
        -------
        (row_copy, animation)
            `row_copy` is the VGroup you can keep on screen after the anim.
        """
        if target_position is None:
            target_position = self.get_right() + RIGHT * 1.5

        row_copy = self._rows[row_idx].copy()
        row_copy.set_opacity(1.0)

        anim = row_copy.animate(run_time=run_time, rate_func=rate_functions.ease_in_out_sine) \
                       .move_to(target_position)
        return row_copy, anim

    def lookup_sequence(
        self,
        row_idx: int,
        target_position=None,
        run_time: float = DEFAULT_RUN_TIME,
    ) -> tuple[VGroup, Succession]:
        """
        Convenience wrapper:  dim_all → highlight_row → extract_row
        Returns (extracted_row_copy, full_succession_animation).
        """
        row_copy, extract_anim = self.extract_row(row_idx, target_position, run_time)
        seq = Succession(
            self.dim_all_rows(),
            self.highlight_row(row_idx, run_time),
            extract_anim,
        )
        return row_copy, seq


# ===========================================================================
# 2.  TokenToVectorAnim
# ===========================================================================

class TokenToVectorAnim(VGroup):
    """
    Three-part visual:
        [TokenBox]  ──arrow──>  [EmbeddingMatrix]  ──arrow──>  [VectorDisplay]

    Call `build_animation()` to get a Succession that plays the whole thing.

    Parameters
    ----------
    token_text : str   — word to show in the token box
    token_id   : int   — row index into the embedding matrix
    vocab_size : int   — forwarded to EmbeddingMatrix
    d_model    : int   — forwarded to EmbeddingMatrix
    """

    def __init__(
        self,
        token_text: str = "king",
        token_id: int   = 4,
        vocab_size: int = 32,
        d_model: int    = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.token_text = token_text
        self.token_id   = token_id

        # ── token box ────────────────────────────────────────────────────────
        self.token_box  = _make_token_box(token_text, token_id)

        # ── embedding matrix ─────────────────────────────────────────────────
        self.emb_matrix = EmbeddingMatrix(
            vocab_size=vocab_size, d_model=d_model, show_values=False
        )
        self.emb_matrix.next_to(self.token_box, RIGHT, buff=1.2)

        # ── vector display ───────────────────────────────────────────────────
        fake_vector = self.emb_matrix._values[token_id % vocab_size]
        self.vec_display = VectorDisplay(fake_vector)
        self.vec_display.next_to(self.emb_matrix, RIGHT, buff=1.2)

        # ── arrows ───────────────────────────────────────────────────────────
        self.arrow_left  = Arrow(
            self.token_box.get_right(),
            self.emb_matrix.get_left(),
            buff=0.1, color=COLORS["arrow"],
        )
        self.arrow_right = Arrow(
            self.emb_matrix.get_right(),
            self.vec_display.get_left(),
            buff=0.1, color=COLORS["embedding"],
        )

        self.add(
            self.token_box, self.emb_matrix, self.vec_display,
            self.arrow_left, self.arrow_right,
        )

    # ── animations ───────────────────────────────────────────────────────────

    def build_animation(self) -> Succession:
        """Full pipeline animation: token → lookup → vector."""
        _, lookup_anim = self.emb_matrix.lookup_sequence(
            self.token_id,
            target_position=self.vec_display.get_center(),
        )
        return Succession(
            # 1) Show token box
            FadeIn(self.token_box, shift=UP * 0.2),
            # 2) Grow arrow to matrix
            GrowArrow(self.arrow_left),
            # 3) Perform lookup
            lookup_anim,
            # 4) Grow arrow to vector
            GrowArrow(self.arrow_right),
            # 5) Reveal vector display
            self.vec_display.appear_animation(),
        )


# ===========================================================================
# 3.  VectorDisplay
# ===========================================================================

class VectorDisplay(VGroup):
    """
    Horizontal bar split into `len(values)` cells.
    Color intensity encodes value magnitude: red (negative) ↔ blue (positive).

    Parameters
    ----------
    values      : array-like  — dimension values
    label       : str | None  — optional label above the bar
    show_values : bool        — display numeric values inside cells
    max_display : int         — truncate display if len(values) > max_display
    cell_size   : float       — cell width
    """

    def __init__(
        self,
        values,
        label:       str | None = None,
        show_values: bool  = True,
        max_display: int   = 16,
        cell_size:   float = CELL_SIZE,
        **kwargs,
    ):
        super().__init__(**kwargs)

        values = list(values)
        self._values = values
        truncated = len(values) > max_display
        display_vals = values[:max_display]

        self._cells: list[Rectangle] = []

        for i, v in enumerate(display_vals):
            rect = Rectangle(
                width=cell_size, height=cell_size * 1.8,
                fill_color=_value_to_color(v),
                fill_opacity=0.8,
                stroke_color=WHITE,
                stroke_width=0.6,
            )
            rect.move_to(_np.array([i * cell_size, 0, 0]))
            self._cells.append(rect)
            self.add(rect)

            if show_values:
                num = Text(f"{v:.2f}", font="Monospace").scale(0.2).set_color(WHITE)
                num.move_to(rect.get_center())
                self.add(num)

        if truncated:
            ellipsis = Text("…", font="Monospace").scale(0.5).set_color(GRAY)
            ellipsis.next_to(self._cells[-1], RIGHT, buff=0.1)
            self.add(ellipsis)
            dim_note = _dim_label(f"d={len(values)}")
            dim_note.next_to(ellipsis, RIGHT, buff=0.15)
            self.add(dim_note)

        # optional label
        if label:
            lbl = Text(label, font="Monospace").scale(LABEL_SCALE + 0.05) \
                      .set_color(COLORS["embedding"]) \
                      .next_to(self, UP, buff=0.15)
            self.add(lbl)

    # ── animations ───────────────────────────────────────────────────────────

    def appear_animation(self, run_time: float = DEFAULT_RUN_TIME) -> LaggedStart:
        """Cells pop in left-to-right."""
        return LaggedStart(
            *[FadeIn(c, shift=DOWN * 0.15) for c in self._cells],
            lag_ratio=0.07,
            run_time=run_time,
        )

    def pulse(self, color: str = COLORS["highlight"], run_time: float = 0.6) -> AnimationGroup:
        """Briefly flash all cells."""
        return AnimationGroup(
            *[Flash(c, color=color, line_length=0.1) for c in self._cells],
            lag_ratio=0.0,
            run_time=run_time,
        )

    def highlight_dims(self, indices: list[int], run_time: float = 0.5) -> AnimationGroup:
        """Highlight specific dimension indices."""
        anims = []
        for i, cell in enumerate(self._cells):
            if i in indices:
                anims.append(
                    cell.animate(run_time=run_time)
                        .set_stroke(color=COLORS["highlight"], width=3)
                        .set_fill(opacity=1.0)
                )
            else:
                anims.append(cell.animate(run_time=run_time).set_opacity(0.3))
        return AnimationGroup(*anims, lag_ratio=0.0)


# ===========================================================================
# 4.  VectorSpaceViz
# ===========================================================================

class VectorSpaceViz(VGroup):
    """
    2D scatter plot of token embeddings projected into 2D.

    Parameters
    ----------
    tokens    : list[str]         — token labels
    positions : list[tuple[float,float]]  — 2D coords for each token
                (use PCA/t-SNE in preprocessing; this just plots them)
    show_axes : bool              — whether to draw x/y axes
    """

    def __init__(
        self,
        tokens:    list[str]               = None,
        positions: list[tuple[float,float]] = None,
        show_axes: bool = True,
        axis_range: float = 4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Defaults: classic word-analogy demo
        if tokens is None:
            tokens = ["king", "queen", "man", "woman", "prince", "princess"]
        if positions is None:
            positions = [
                ( 1.8,  1.5),
                ( 1.8, -1.0),
                (-1.5,  1.5),
                (-1.5, -1.0),
                ( 0.6,  2.8),
                ( 0.6,  0.0),
            ]

        self._tokens    = tokens
        self._positions = positions
        self._dots: dict[str, Dot] = {}
        self._labels: dict[str, Text] = {}

        # ── axes ─────────────────────────────────────────────────────────────
        if show_axes:
            axes = Axes(
                x_range=[-axis_range, axis_range, 1],
                y_range=[-axis_range, axis_range, 1],
                x_length=axis_range * 1.6,
                y_length=axis_range * 1.6,
                axis_config={"color": COLORS["dim"], "stroke_width": 1},
                tips=False,
            )
            self.add(axes)
            self._axes = axes
        else:
            self._axes = None

        # ── dots + labels ─────────────────────────────────────────────────────
        color_cycle = [
            COLORS["token"], COLORS["embedding"], COLORS["attention"],
            COLORS["ffn"], COLORS["query"], COLORS["value"],
        ]
        for idx, (tok, pos) in enumerate(zip(tokens, positions)):
            color = color_cycle[idx % len(color_cycle)]
            dot = Dot(point=_np.array([pos[0], pos[1], 0]), color=color, radius=0.12)
            lbl = Text(tok, font="Monospace") \
                      .scale(LABEL_SCALE) \
                      .set_color(color) \
                      .next_to(dot, UP + RIGHT, buff=0.05)
            self._dots[tok]   = dot
            self._labels[tok] = lbl
            self.add(dot, lbl)

    # ── animations ───────────────────────────────────────────────────────────

    def appear_animation(self, run_time: float = 1.5) -> LaggedStart:
        """Dots and labels fade in staggered."""
        items = []
        for tok in self._tokens:
            items.append(FadeIn(self._dots[tok], scale=0.4))
            items.append(Write(self._labels[tok]))
        return LaggedStart(*items, lag_ratio=0.15, run_time=run_time)

    def draw_distance(
        self,
        tok_a: str,
        tok_b: str,
        label: str | None = None,
        color: str = COLORS["highlight"],
        run_time: float = DEFAULT_RUN_TIME,
    ) -> tuple[VGroup, Animation]:
        """
        Draw a dashed line between two token points with an optional distance label.
        Returns (line_group, animation).
        """
        dot_a = self._dots[tok_a]
        dot_b = self._dots[tok_b]
        line = Line(
            dot_a.get_center(), dot_b.get_center(),
            color=color, stroke_width=1.5,
        ).set_stroke(dash_array=[0.08, 0.08])

        group = VGroup(line)
        if label:
            mid = (dot_a.get_center() + dot_b.get_center()) / 2
            lbl = Text(label, font="Monospace") \
                      .scale(LABEL_SCALE) \
                      .set_color(color) \
                      .move_to(mid + UP * 0.2)
            group.add(lbl)

        return group, Create(group, run_time=run_time)

    def draw_analogy_arrow(
        self,
        tok_start: str,
        tok_end:   str,
        label:     str = "",
        color:     str = COLORS["query"],
        run_time:  float = DEFAULT_RUN_TIME,
    ) -> tuple[Arrow, Animation]:
        """
        Draw an arrow from one token point to another (e.g., man → woman).
        Returns (arrow, animation).
        """
        start = self._dots[tok_start].get_center()
        end   = self._dots[tok_end].get_center()
        arrow = Arrow(start, end, buff=0.12, color=color, stroke_width=2)

        if label:
            mid = (start + end) / 2
            lbl = Text(label, font="Monospace") \
                      .scale(LABEL_SCALE) \
                      .set_color(color) \
                      .next_to(arrow, UP, buff=0.08)
            group = VGroup(arrow, lbl)
            return group, GrowArrow(arrow)

        return arrow, GrowArrow(arrow)

    def highlight_cluster(
        self,
        token_list: list[str],
        color: str = COLORS["highlight"],
        run_time: float = 0.6,
    ) -> AnimationGroup:
        """Flash a subset of token dots to draw attention to a cluster."""
        anims = [
            Flash(self._dots[tok], color=color, line_length=0.18)
            for tok in token_list if tok in self._dots
        ]
        return AnimationGroup(*anims, lag_ratio=0.0, run_time=run_time)


# ===========================================================================
# 5.  EmbeddingAddition
# ===========================================================================

class EmbeddingAddition(VGroup):
    """
    Visualizes element-wise addition of two vectors (e.g., token emb + pos enc).

    Layout (vertical):
        ┌──────────────────┐   ← vec_a  (e.g. token embedding)
        ┌──────────────────┐   ← vec_b  (e.g. positional encoding)
        ──────────────────     ← plus sign
        ┌──────────────────┐   ← result

    Parameters
    ----------
    vec_a    : array-like — first vector values
    vec_b    : array-like — second vector values
    label_a  : str        — label for first vector
    label_b  : str        — label for second vector
    label_out: str        — label for result vector
    """

    def __init__(
        self,
        vec_a,
        vec_b,
        label_a:   str = "Token Embedding",
        label_b:   str = "Positional Encoding",
        label_out: str = "Input to Transformer",
        **kwargs,
    ):
        super().__init__(**kwargs)

        vec_a   = _np.array(vec_a, dtype=float)
        vec_b   = _np.array(vec_b, dtype=float)
        result  = vec_a + vec_b
        # clip result for display purposes
        result  = _np.clip(result, -1, 1)

        self._vec_a_disp   = VectorDisplay(vec_a,   label=label_a,   show_values=True)
        self._vec_b_disp   = VectorDisplay(vec_b,   label=label_b,   show_values=True)
        self._result_disp  = VectorDisplay(result,  label=label_out, show_values=True)

        # plus sign
        plus = MathTex("+").scale(1.2).set_color(WHITE)

        # separator line
        width = self._vec_a_disp.width + 0.4
        sep   = Line(LEFT * width / 2, RIGHT * width / 2, color=GRAY, stroke_width=1)

        # vertical layout
        self._vec_a_disp.move_to(ORIGIN)
        plus.next_to(self._vec_a_disp, DOWN, buff=0.25)
        self._vec_b_disp.next_to(plus, DOWN, buff=0.25)
        sep.next_to(self._vec_b_disp, DOWN, buff=0.2)
        self._result_disp.next_to(sep, DOWN, buff=0.2)

        self.add(
            self._vec_a_disp, plus,
            self._vec_b_disp, sep,
            self._result_disp,
        )

        self._plus = plus
        self._sep  = sep

    # ── animations ───────────────────────────────────────────────────────────

    def build_animation(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """
        Step-by-step:
          1) Show vec_a
          2) Show plus + vec_b
          3) Show separator
          4) Cells of result appear one-by-one (like summing)
        """
        cell_anims = [
            FadeIn(cell, shift=DOWN * 0.1)
            for cell in self._result_disp._cells
        ]
        return Succession(
            self._vec_a_disp.appear_animation(run_time),
            AnimationGroup(
                FadeIn(self._plus),
                self._vec_b_disp.appear_animation(run_time),
                lag_ratio=0.0,
            ),
            Create(self._sep),
            LaggedStart(*cell_anims, lag_ratio=0.1, run_time=run_time),
        )

    def highlight_column(self, col_idx: int, run_time: float = 0.5) -> AnimationGroup:
        """
        Flash the same column index across all three vectors to show
        how a single dimension adds.
        """
        targets = []
        for vec_disp in (self._vec_a_disp, self._vec_b_disp, self._result_disp):
            if col_idx < len(vec_disp._cells):
                targets.append(
                    Indicate(vec_disp._cells[col_idx], color=COLORS["highlight"])
                )
        return AnimationGroup(*targets, lag_ratio=0.0, run_time=run_time)


# ===========================================================================
# Helper: minimal TokenBox  (avoids circular import from tokenization/)
# ===========================================================================

def _make_token_box(text: str, token_id: int) -> VGroup:
    """Lightweight token box for use inside this module."""
    rect = Rectangle(
        width=1.2, height=0.6,
        fill_color=COLORS["token"], fill_opacity=0.85,
        stroke_color=WHITE, stroke_width=1.2,
        corner_radius=0.08,
    )
    word_lbl = Text(text, font="Monospace") \
                   .scale(0.38) \
                   .set_color(WHITE) \
                   .move_to(rect.get_center())
    id_lbl   = Text(f"id={token_id}", font="Monospace") \
                   .scale(0.26) \
                   .set_color(LIGHT_GRAY) \
                   .next_to(rect, DOWN, buff=0.08)
    return VGroup(rect, word_lbl, id_lbl)