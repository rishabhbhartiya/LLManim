"""
llmanim/attention/qkv_projection.py
=============================================
Reusable Manim assets for visualizing the QKV projection step
in Transformer attention.

Assets
------
InputMatrix          — token vectors stacked as rows  (seq_len × d_model)
WeightMatrix         — projection weight grid         (d_model × d_k)
ProjectionAnim       — full matmul walk-through scene
QKVDisplay           — Q, K, V vectors side by side
LinearLayerAnim      — node-graph + matrix form of one linear layer

Usage (quick-start)
-------------------
    from llmanim.attention.qkv_projection import (
        InputMatrix, WeightMatrix, ProjectionAnim, QKVDisplay, LinearLayerAnim,
    )

    class MyScene(Scene):
        def construct(self):
            proj = ProjectionAnim(seq_len=4, d_model=6, d_k=4)
            self.play(*proj.build())
            self.wait()

Color convention (mirrors styles/colors.py)
-------------------------------------------
    query  (Q)  →  #E91E63   (pink)
    key    (K)  →  #3F51B5   (indigo)
    value  (V)  →  #009688   (teal)
    weight      →  #795548   (brown)
    token       →  #4CAF50   (green)
    highlight   →  #FFEB3B   (yellow)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from manim import (
    BLUE, DOWN, GREEN, LEFT, ORIGIN, PI, RIGHT, UP, WHITE, YELLOW,
    Animation, AnimationGroup, Arrow, Brace, Create, FadeIn, FadeOut,
    Flash, LaggedStart, MathTex, Rectangle, Scene, Succession, Text,
    Transform, VGroup, Write,
    always_redraw, linear,
)
from manim import ManimColor as MColor
from manim import (
    BLACK, DARK_GRAY, GRAY, LIGHT_GRAY,
    Dot, Line, SurroundingRectangle,
    Circle, Square,
)

# ---------------------------------------------------------------------------
# Inline color palette (so this file works standalone without styles/)
# ---------------------------------------------------------------------------
_C = {
    "query":        MColor("#E91E63"),
    "key":          MColor("#3F51B5"),
    "value":        MColor("#009688"),
    "weight":       MColor("#795548"),
    "token":        MColor("#4CAF50"),
    "highlight":    MColor("#FFEB3B"),
    "arrow":        MColor("#FFFFFF"),
    "dim":          MColor("#555555"),
    "attention":    MColor("#9C27B0"),
    "bg_cell":      MColor("#1A1A2E"),
    "grid_stroke":  MColor("#333355"),
}

_QKV_COLORS = {"Q": _C["query"], "K": _C["key"], "V": _C["value"]}
_QKV_W_LABELS = {"Q": "W_Q", "K": "W_K", "V": "W_V"}


# ============================================================
# Helper utilities (keep internal — use utils.py in real lib)
# ============================================================

def _make_cell(
    width: float,
    height: float,
    fill_color=_C["bg_cell"],
    stroke_color=_C["grid_stroke"],
    fill_opacity: float = 0.85,
) -> Rectangle:
    return Rectangle(
        width=width,
        height=height,
        fill_color=fill_color,
        fill_opacity=fill_opacity,
        stroke_color=stroke_color,
        stroke_width=1.0,
    )


def _grid(
    rows: int,
    cols: int,
    cell_w: float = 0.35,
    cell_h: float = 0.35,
    fill_color=_C["bg_cell"],
    stroke_color=_C["grid_stroke"],
) -> VGroup:
    """Return a VGroup of Rectangle cells arranged in a rows×cols grid."""
    group = VGroup()
    for r in range(rows):
        for c in range(cols):
            cell = _make_cell(cell_w, cell_h, fill_color, stroke_color)
            cell.move_to(
                np.array([c * cell_w - (cols - 1) * cell_w / 2,
                          -r * cell_h + (rows - 1) * cell_h / 2,
                          0])
            )
            group.add(cell)
    return group


def _dim_brace(mob, direction, label: str, font_size: int = 20, color=WHITE):
    """Return (Brace, Text) pair for a dimension annotation."""
    brace = Brace(mob, direction=direction, color=color)
    text = brace.get_tex(label, buff=0.1)
    text.set_color(color).scale(font_size / 28)
    return brace, text


# ============================================================
# 1. InputMatrix
# ============================================================

class InputMatrix(VGroup):
    """
    Visual representation of the input token matrix (seq_len × d_model).

    Parameters
    ----------
    seq_len : int   — number of tokens (rows)
    d_model : int   — embedding dimension (columns)
    cell_w, cell_h  — cell size in Manim units
    token_labels    — optional list of strings shown on the left

    Public methods
    --------------
    highlight_row(row_idx)        → Animation  — highlight one token row
    dehighlight_row(row_idx)      → Animation
    extract_row_anim(row_idx)     → (Animation, VGroup)  — slides row out
    build_in()                    → AnimationGroup — creation animation
    """

    def __init__(
        self,
        seq_len: int = 4,
        d_model: int = 8,
        cell_w: float = 0.32,
        cell_h: float = 0.40,
        token_labels: Sequence[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.cell_w = cell_w
        self.cell_h = cell_h

        # ---- grid ----
        self.cells = _grid(seq_len, d_model, cell_w, cell_h)
        self.add(self.cells)

        # ---- row colour strips (token colours) ----
        _token_palette = [
            MColor("#4CAF50"), MColor("#2196F3"), MColor("#FF9800"),
            MColor("#9C27B0"), MColor("#F44336"), MColor("#00BCD4"),
        ]
        self._row_overlays = VGroup()
        for r in range(seq_len):
            color = _token_palette[r % len(_token_palette)]
            overlay = Rectangle(
                width=cell_w * d_model,
                height=cell_h,
                fill_color=color,
                fill_opacity=0.0,        # invisible until highlighted
                stroke_width=0,
            )
            overlay.move_to(self.cells[r * d_model].get_center()
                            + RIGHT * cell_w * (d_model - 1) / 2)
            self._row_overlays.add(overlay)
        self.add(self._row_overlays)

        # ---- outer border ----
        self.border = SurroundingRectangle(
            self.cells, color=_C["token"], buff=0.04, stroke_width=1.5
        )
        self.add(self.border)

        # ---- dimension braces ----
        b_rows, t_rows = _dim_brace(self.cells, LEFT, str(seq_len))
        b_cols, t_cols = _dim_brace(self.cells, UP, str(d_model))
        self._braces = VGroup(b_rows, t_rows, b_cols, t_cols)
        self.add(self._braces)

        # ---- token labels ----
        self._token_labels = VGroup()
        if token_labels:
            for r, lbl in enumerate(token_labels[:seq_len]):
                cell_center = self.cells[r * d_model].get_center()
                t = Text(lbl, font_size=16, color=_C["token"])
                t.next_to(cell_center, LEFT, buff=0.55)
                self._token_labels.add(t)
            self.add(self._token_labels)

        # ---- matrix label ----
        self.label = MathTex(r"\mathbf{X}", font_size=32, color=_C["token"])
        self.label.next_to(self.border, DOWN, buff=0.25)
        self.add(self.label)

    # ------------------------------------------------------------------
    # Row utilities
    # ------------------------------------------------------------------

    def _row_cells(self, row_idx: int) -> VGroup:
        start = row_idx * self.d_model
        return VGroup(*self.cells[start: start + self.d_model])

    def highlight_row(self, row_idx: int, color=None) -> Animation:
        overlay = self._row_overlays[row_idx]
        target = overlay.copy()
        target.set_fill(
            color or _C["highlight"], opacity=0.45
        )
        return Transform(overlay, target, run_time=0.4)

    def dehighlight_row(self, row_idx: int) -> Animation:
        overlay = self._row_overlays[row_idx]
        target = overlay.copy()
        target.set_fill(opacity=0.0)
        return Transform(overlay, target, run_time=0.3)

    def extract_row_anim(
        self, row_idx: int, target_position=None
    ) -> tuple[Animation, VGroup]:
        """
        Returns (animation, extracted_row_copy).
        The copy travels to target_position (default: RIGHT of the matrix).
        """
        row_copy = self._row_cells(row_idx).copy()
        row_copy.set_fill(_C["highlight"], opacity=0.9)
        if target_position is None:
            target_position = self.get_right() + RIGHT * 1.5
        row_copy.generate_target()
        row_copy.target.move_to(target_position)
        from manim import MoveToTarget
        return MoveToTarget(row_copy, run_time=0.8, path_arc=0.3), row_copy

    def build_in(self) -> AnimationGroup:
        return LaggedStart(
            *[FadeIn(self.cells[r * self.d_model: (r + 1) * self.d_model],
                     shift=RIGHT * 0.1)
              for r in range(self.seq_len)],
            lag_ratio=0.15,
            run_time=1.2,
        )


# ============================================================
# 2. WeightMatrix
# ============================================================

class WeightMatrix(VGroup):
    """
    Weight matrix W_Q / W_K / W_V  (d_model × d_k).

    Parameters
    ----------
    kind         : 'Q' | 'K' | 'V'
    d_model, d_k : dimensions
    cell_w, cell_h

    Public methods
    --------------
    highlight_col(col_idx)    → Animation
    dehighlight_col(col_idx)  → Animation
    pulse()                   → Animation  — brief glow
    build_in()                → AnimationGroup
    """

    def __init__(
        self,
        kind: str = "Q",
        d_model: int = 8,
        d_k: int = 4,
        cell_w: float = 0.32,
        cell_h: float = 0.32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert kind in ("Q", "K", "V"), "kind must be 'Q', 'K', or 'V'"
        self.kind = kind
        self.d_model = d_model
        self.d_k = d_k
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.color = _QKV_COLORS[kind]

        # ---- grid ----
        self.cells = _grid(d_model, d_k, cell_w, cell_h,
                           fill_color=_C["bg_cell"],
                           stroke_color=self.color)
        # tint the whole grid with the kind color
        for cell in self.cells:
            cell.set_fill(self.color, opacity=0.18)
        self.add(self.cells)

        # ---- column overlays for highlighting ----
        self._col_overlays = VGroup()
        for c in range(d_k):
            overlay = Rectangle(
                width=cell_w,
                height=cell_h * d_model,
                fill_color=self.color,
                fill_opacity=0.0,
                stroke_width=0,
            )
            # centre the overlay on the column
            first_cell = self.cells[c]   # row=0, col=c
            last_cell = self.cells[(d_model - 1) * d_k + c]
            overlay.move_to(
                (first_cell.get_center() + last_cell.get_center()) / 2
            )
            self._col_overlays.add(overlay)
        self.add(self._col_overlays)

        # ---- border ----
        self.border = SurroundingRectangle(
            self.cells, color=self.color, buff=0.04, stroke_width=2
        )
        self.add(self.border)

        # ---- dimension braces ----
        b_rows, t_rows = _dim_brace(self.cells, LEFT, str(d_model), color=self.color)
        b_cols, t_cols = _dim_brace(self.cells, UP, str(d_k), color=self.color)
        self._braces = VGroup(b_rows, t_rows, b_cols, t_cols)
        self.add(self._braces)

        # ---- matrix label ----
        tex_label = _QKV_W_LABELS[kind]
        self.label = MathTex(
            rf"\mathbf{{{tex_label}}}", font_size=32, color=self.color
        )
        self.label.next_to(self.border, DOWN, buff=0.25)
        self.add(self.label)

    # ------------------------------------------------------------------
    def _col_cells(self, col_idx: int) -> VGroup:
        return VGroup(*[self.cells[r * self.d_k + col_idx]
                        for r in range(self.d_model)])

    def highlight_col(self, col_idx: int) -> Animation:
        overlay = self._col_overlays[col_idx]
        target = overlay.copy().set_fill(self.color, opacity=0.55)
        return Transform(overlay, target, run_time=0.35)

    def dehighlight_col(self, col_idx: int) -> Animation:
        overlay = self._col_overlays[col_idx]
        target = overlay.copy().set_fill(opacity=0.0)
        return Transform(overlay, target, run_time=0.28)

    def pulse(self) -> Animation:
        return Flash(
            self.border, color=self.color, flash_radius=0.25,
            line_length=0.15, num_lines=12, run_time=0.6,
        )

    def build_in(self) -> AnimationGroup:
        return LaggedStart(
            FadeIn(self.cells, shift=DOWN * 0.1),
            FadeIn(self.border),
            FadeIn(self.label),
            lag_ratio=0.2,
            run_time=0.9,
        )


# ============================================================
# 3. QKVDisplay
# ============================================================

class QKVDisplay(VGroup):
    """
    Shows Q, K, V output vectors side by side after projection.

    Parameters
    ----------
    d_k         — dimension of each vector (number of cells)
    values_q/k/v — optional float arrays of length d_k for cell shading
    spacing     — horizontal gap between the three vectors

    Public methods
    --------------
    appear()            → AnimationGroup — all three fade/grow in at once
    highlight(kind)     → Animation      — glow one vector ('Q'|'K'|'V')
    show_values(kind, values) → AnimationGroup — shade cells by value
    """

    def __init__(
        self,
        d_k: int = 4,
        values_q=None,
        values_k=None,
        values_v=None,
        cell_h: float = 0.36,
        cell_w: float = 0.52,
        spacing: float = 0.80,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_k = d_k
        self.cell_h = cell_h
        self.cell_w = cell_w

        self._vectors: dict[str, VGroup] = {}
        self._borders: dict[str, SurroundingRectangle] = {}
        self._labels: dict[str, MathTex] = {}

        total_w = cell_w + spacing
        offsets = {"Q": -total_w, "K": 0.0, "V": total_w}

        for kind in ("Q", "K", "V"):
            color = _QKV_COLORS[kind]
            # column of cells
            vec = VGroup()
            for i in range(d_k):
                cell = _make_cell(cell_w, cell_h,
                                  fill_color=_C["bg_cell"],
                                  stroke_color=color)
                cell.set_fill(color, opacity=0.20)
                cell.move_to(UP * (d_k / 2 - 0.5 - i) * cell_h
                             + RIGHT * offsets[kind])
                vec.add(cell)
            self._vectors[kind] = vec
            self.add(vec)

            # border
            border = SurroundingRectangle(
                vec, color=color, buff=0.06, stroke_width=2
            )
            self._borders[kind] = border
            self.add(border)

            # label
            lbl = MathTex(
                rf"\mathbf{{{kind}}}", font_size=36, color=color
            )
            lbl.next_to(border, UP, buff=0.20)
            self._labels[kind] = lbl
            self.add(lbl)

        # apply initial value shading
        for kind, vals in (("Q", values_q), ("K", values_k), ("V", values_v)):
            if vals is not None:
                self._shade_cells(kind, vals)

    # ------------------------------------------------------------------
    def _shade_cells(self, kind: str, values):
        color = _QKV_COLORS[kind]
        arr = np.array(values, dtype=float)
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        for cell, v in zip(self._vectors[kind], arr_norm):
            cell.set_fill(color, opacity=float(0.15 + 0.70 * v))

    def appear(self) -> AnimationGroup:
        anims = []
        for kind in ("Q", "K", "V"):
            anims += [
                FadeIn(self._vectors[kind], shift=UP * 0.15),
                FadeIn(self._borders[kind]),
                Write(self._labels[kind]),
            ]
        return LaggedStart(*anims, lag_ratio=0.12, run_time=1.4)

    def highlight(self, kind: str) -> Animation:
        border = self._borders[kind]
        return Flash(
            border, color=_QKV_COLORS[kind],
            flash_radius=0.20, num_lines=10, run_time=0.5,
        )

    def show_values(self, kind: str, values) -> AnimationGroup:
        color = _QKV_COLORS[kind]
        arr = np.array(values, dtype=float)
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        anims = []
        for cell, v in zip(self._vectors[kind], arr_norm):
            target = cell.copy().set_fill(color, opacity=float(0.15 + 0.70 * v))
            anims.append(Transform(cell, target, run_time=0.3))
        return LaggedStart(*anims, lag_ratio=0.06)


# ============================================================
# 4. LinearLayerAnim  (node graph + matrix form)
# ============================================================

class LinearLayerAnim(VGroup):
    """
    Visualises a single linear projection as both:
      • a bipartite node graph (input nodes → output nodes with edges)
      • the equivalent matrix equation shown alongside

    Parameters
    ----------
    kind         : 'Q' | 'K' | 'V'
    n_in         : number of input nodes shown  (d_model, capped for clarity)
    n_out        : number of output nodes shown (d_k, capped for clarity)
    node_radius  : radius of each node

    Public methods
    --------------
    build_in()                → Succession of Animations
    forward_pass_pulse()      → Animation  — wave of colour through edges
    show_matrix_equation()    → AnimationGroup
    """

    def __init__(
        self,
        kind: str = "Q",
        n_in: int = 6,
        n_out: int = 4,
        node_radius: float = 0.14,
        h_spacing: float = 2.20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kind = kind
        self.n_in = n_in
        self.n_out = n_out
        self.color = _QKV_COLORS[kind]
        self.node_radius = node_radius

        v_in = 0.42 * (n_in - 1)
        v_out = 0.42 * (n_out - 1)

        # ---- input nodes ----
        self._in_nodes = VGroup()
        for i in range(n_in):
            d = Dot(radius=node_radius, color=_C["token"])
            d.move_to(LEFT * (h_spacing / 2)
                      + UP * (v_in / 2 - i * (v_in / max(n_in - 1, 1))))
            self._in_nodes.add(d)
        self.add(self._in_nodes)

        # ---- output nodes ----
        self._out_nodes = VGroup()
        for j in range(n_out):
            d = Dot(radius=node_radius, color=self.color)
            d.move_to(RIGHT * (h_spacing / 2)
                      + UP * (v_out / 2 - j * (v_out / max(n_out - 1, 1))))
            self._out_nodes.add(d)
        self.add(self._out_nodes)

        # ---- edges ----
        self._edges = VGroup()
        for in_node in self._in_nodes:
            for out_node in self._out_nodes:
                edge = Line(
                    in_node.get_center(),
                    out_node.get_center(),
                    stroke_color=_C["weight"],
                    stroke_width=0.8,
                    stroke_opacity=0.55,
                )
                self._edges.add(edge)
        # insert edges behind nodes
        self.submobjects.insert(0, self._edges)

        # ---- input / output labels ----
        lbl_in = Text("input\n(d_model)", font_size=14, color=_C["token"])
        lbl_in.next_to(self._in_nodes, LEFT, buff=0.25)
        lbl_out = Text(f"output\n(d_k)", font_size=14, color=self.color)
        lbl_out.next_to(self._out_nodes, RIGHT, buff=0.25)
        self.add(lbl_in, lbl_out)

        # ---- weight matrix equation (hidden until show_matrix_equation) ----
        tex = (rf"\mathbf{{X}} \cdot \mathbf{{W_{{{self.kind}}}}}"
               rf" = \mathbf{{{self.kind}}}")
        self._equation = MathTex(tex, font_size=28, color=self.color)
        self._equation.next_to(self, DOWN, buff=0.45)
        self._equation.set_opacity(0)
        self.add(self._equation)

    # ------------------------------------------------------------------
    def build_in(self) -> Succession:
        from manim import GrowFromCenter
        return Succession(
            LaggedStart(
                *[GrowFromCenter(n) for n in self._in_nodes],
                lag_ratio=0.08, run_time=0.6,
            ),
            LaggedStart(
                *[Create(e) for e in self._edges],
                lag_ratio=0.02, run_time=0.9,
            ),
            LaggedStart(
                *[GrowFromCenter(n) for n in self._out_nodes],
                lag_ratio=0.08, run_time=0.6,
            ),
        )

    def forward_pass_pulse(self) -> AnimationGroup:
        """Colour wave: each input node pulses and sends a flash along its edges."""
        anims = []
        for i, in_node in enumerate(self._in_nodes):
            anims.append(
                Flash(in_node, color=self.color,
                      flash_radius=self.node_radius * 2.5,
                      num_lines=6, run_time=0.4)
            )
        for out_node in self._out_nodes:
            anims.append(
                Flash(out_node, color=self.color,
                      flash_radius=self.node_radius * 2.5,
                      num_lines=6, run_time=0.35)
            )
        return LaggedStart(*anims, lag_ratio=0.10, run_time=1.2)

    def show_matrix_equation(self) -> AnimationGroup:
        target = self._equation.copy().set_opacity(1)
        return AnimationGroup(
            Transform(self._equation, target, run_time=0.6),
            Flash(self._equation.copy().set_opacity(1),
                  color=self.color, flash_radius=0.4, run_time=0.5),
        )


# ============================================================
# 5. ProjectionAnim  — orchestrates the full matmul walk-through
# ============================================================

class ProjectionAnim:
    """
    High-level scene helper that orchestrates:
      InputMatrix  ×  WeightMatrix  →  output row (one token at a time)

    This is NOT a VGroup — it holds Manim objects and yields animation
    sequences you call with scene.play().

    Parameters
    ----------
    seq_len, d_model, d_k  — dimensions
    kind                   — 'Q' | 'K' | 'V'  (default runs all three)
    token_labels           — list of token strings for rows

    Usage
    -----
        pa = ProjectionAnim(seq_len=4, d_model=6, d_k=4, kind='Q')
        for anim_group in pa.iter_animations():
            scene.play(*anim_group)
            scene.wait(0.3)
    """

    def __init__(
        self,
        seq_len: int = 4,
        d_model: int = 8,
        d_k: int = 4,
        kind: str = "Q",
        token_labels: Sequence[str] | None = None,
    ):
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_k = d_k
        self.kind = kind
        self.color = _QKV_COLORS[kind]

        # ---- build sub-objects ----
        self.input_mat = InputMatrix(
            seq_len=seq_len,
            d_model=d_model,
            cell_w=0.28,
            cell_h=0.36,
            token_labels=token_labels,
        )
        self.weight_mat = WeightMatrix(
            kind=kind,
            d_model=d_model,
            d_k=d_k,
            cell_w=0.28,
            cell_h=0.28,
        )
        self.qkv_display = QKVDisplay(d_k=seq_len, cell_h=0.36)

        # ---- position the objects ----
        # X (left) — W (centre) — output (right)
        self.input_mat.move_to(LEFT * 4.2)
        self.weight_mat.move_to(ORIGIN)
        self.qkv_display.move_to(RIGHT * 4.2)

        # ---- multiplication arrow and equals sign ----
        self._times = MathTex(r"\times", font_size=40, color=WHITE)
        self._times.move_to(
            (self.input_mat.get_right() + self.weight_mat.get_left()) / 2
        )
        self._equals = MathTex(r"=", font_size=40, color=WHITE)
        self._equals.move_to(
            (self.weight_mat.get_right() + self.qkv_display.get_left()) / 2
        )

    # ------------------------------------------------------------------
    def all_objects(self) -> list:
        return [
            self.input_mat, self.weight_mat, self.qkv_display,
            self._times, self._equals,
        ]

    def build_in_animations(self) -> list[Animation]:
        """Return list of animations to place all objects on screen."""
        return [
            self.input_mat.build_in(),
            self.weight_mat.build_in(),
            FadeIn(self._times),
            FadeIn(self._equals),
            self.qkv_display.appear(),
        ]

    def iter_row_highlight_animations(self):
        """
        Generator — yields one (row_highlight, col_highlight, dehighlight)
        tuple per token row, showing which row of X combines with each
        column of W to produce one output row.
        """
        for r in range(self.seq_len):
            row_hi = self.input_mat.highlight_row(r)
            col_anim_list = [
                self.weight_mat.highlight_col(c) for c in range(self.d_k)
            ]
            row_dehi = self.input_mat.dehighlight_row(r)
            col_dehi_list = [
                self.weight_mat.dehighlight_col(c) for c in range(self.d_k)
            ]
            yield row_hi, col_anim_list, row_dehi, col_dehi_list

    def title_label(self) -> MathTex:
        """Return a scene title for this projection kind."""
        tex = (rf"\text{{QKV Projection — }}\mathbf{{W_{{{self.kind}}}}}")
        lbl = MathTex(tex, font_size=30, color=self.color)
        lbl.to_edge(UP, buff=0.25)
        return lbl


# ============================================================
# 6. ThreeProjectionLayout  — Q, K, V side by side
# ============================================================

class ThreeProjectionLayout(VGroup):
    """
    Shows all three projections (Q, K, V) stacked or side by side,
    each as a compact WeightMatrix with label.

    Parameters
    ----------
    d_model, d_k
    orientation : 'horizontal' | 'vertical'

    Public methods
    --------------
    build_in()            → AnimationGroup
    highlight(kind)       → Animation   — glow one matrix
    pulse_all()           → AnimationGroup
    """

    def __init__(
        self,
        d_model: int = 6,
        d_k: int = 4,
        orientation: str = "horizontal",
        spacing: float = 1.20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.matrices: dict[str, WeightMatrix] = {}

        positions = {
            "horizontal": [LEFT * spacing * 2, ORIGIN, RIGHT * spacing * 2],
            "vertical":   [UP * spacing, ORIGIN, DOWN * spacing],
        }[orientation]

        for kind, pos in zip(("Q", "K", "V"), positions):
            wm = WeightMatrix(kind=kind, d_model=d_model, d_k=d_k,
                              cell_w=0.26, cell_h=0.26)
            wm.move_to(pos)
            self.matrices[kind] = wm
            self.add(wm)

        # connecting label
        title = Text(
            "Three learned projections", font_size=22, color=LIGHT_GRAY
        )
        title.next_to(self, UP, buff=0.40)
        self.add(title)

    # ------------------------------------------------------------------
    def build_in(self) -> AnimationGroup:
        return LaggedStart(
            *[m.build_in() for m in self.matrices.values()],
            lag_ratio=0.30,
            run_time=1.5,
        )

    def highlight(self, kind: str) -> Animation:
        return self.matrices[kind].pulse()

    def pulse_all(self) -> AnimationGroup:
        return LaggedStart(
            *[m.pulse() for m in self.matrices.values()],
            lag_ratio=0.20,
            run_time=1.0,
        )


# ============================================================
# 7. Demo Scene  (run with: manim -pql qkv_projection.py DemoQKV)
# ============================================================

class DemoQKV(Scene):
    """
    Quick demo — renders the full QKV projection pipeline.
    Run with:
        manim -pql qkv_projection.py DemoQKV
    """

    def construct(self):
        # --- title ---
        title = Text(
            "QKV Projection", font_size=38,
            color=_C["attention"], weight="BOLD"
        )
        subtitle = Text(
            "How Q, K, V vectors are computed from token embeddings",
            font_size=20, color=LIGHT_GRAY,
        )
        subtitle.next_to(title, DOWN, buff=0.18)
        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.1))
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(subtitle))

        # -------------------------------------------------------
        # Step 1: show the three weight matrices
        # -------------------------------------------------------
        step1 = Text("Step 1 — Three weight matrices", font_size=24,
                     color=WHITE)
        step1.to_edge(UP, buff=0.3)
        self.play(FadeIn(step1, shift=DOWN * 0.1))

        layout = ThreeProjectionLayout(d_model=5, d_k=4, spacing=1.10)
        layout.center()
        self.play(layout.build_in())
        self.wait(0.5)
        self.play(layout.pulse_all())
        self.wait(0.8)
        self.play(FadeOut(layout), FadeOut(step1))

        # -------------------------------------------------------
        # Step 2: one projection in detail (Q)
        # -------------------------------------------------------
        step2 = Text("Step 2 — X · W_Q  =  Q", font_size=24,
                     color=_C["query"])
        step2.to_edge(UP, buff=0.3)
        self.play(FadeIn(step2))

        token_labels = ["[CLS]", "The", "cat", "sat"]
        proj = ProjectionAnim(
            seq_len=4, d_model=6, d_k=4, kind="Q",
            token_labels=token_labels,
        )

        # place everything
        for obj in proj.all_objects():
            self.add(obj)

        for anim in proj.build_in_animations():
            self.play(anim)
        self.wait(0.4)

        # walk through row × column highlights
        for row_hi, col_anims, row_dehi, col_dehis in \
                proj.iter_row_highlight_animations():
            self.play(row_hi)
            self.play(AnimationGroup(*col_anims, lag_ratio=0.12))
            self.wait(0.25)
            self.play(AnimationGroup(row_dehi, *col_dehis))

        self.wait(0.5)
        self.play(*[FadeOut(o) for o in proj.all_objects()], FadeOut(step2))

        # -------------------------------------------------------
        # Step 3: Q K V side by side
        # -------------------------------------------------------
        step3 = Text("Step 3 — Q, K, V vectors produced", font_size=24,
                     color=WHITE)
        step3.to_edge(UP, buff=0.3)
        self.play(FadeIn(step3))

        qkv = QKVDisplay(d_k=6)
        qkv.center()
        self.play(qkv.appear())
        self.wait(0.4)

        for kind in ("Q", "K", "V"):
            self.play(qkv.highlight(kind))
            rng = np.random.default_rng(42 + ord(kind))
            vals = rng.uniform(-1, 1, 6)
            self.play(qkv.show_values(kind, vals))
            self.wait(0.2)

        self.wait(0.5)
        self.play(FadeOut(qkv), FadeOut(step3))

        # -------------------------------------------------------
        # Step 4: node-graph view
        # -------------------------------------------------------
        step4 = Text("Step 4 — Linear layer (node view)", font_size=24,
                     color=WHITE)
        step4.to_edge(UP, buff=0.3)
        self.play(FadeIn(step4))

        for kind in ("Q", "K", "V"):
            lla = LinearLayerAnim(kind=kind, n_in=5, n_out=4)
            lla.center()
            self.play(lla.build_in())
            self.wait(0.3)
            self.play(lla.forward_pass_pulse())
            self.play(lla.show_matrix_equation())
            self.wait(0.6)
            self.play(FadeOut(lla))

        self.play(FadeOut(step4))

        # fin
        fin = Text("QKV Projection  ✓", font_size=40,
                   color=_C["token"], weight="BOLD")
        self.play(Write(fin))
        self.wait(1.5)