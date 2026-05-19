"""
attention/attention_output.py
─────────────────────────────
Reusable Manim assets for the Attention Output stage of a Transformer.

Assets
------
WeightedSumAnim        — animates softmax weights × Value rows → summed output
AttentionOutputVector  — displays the output vector with per-token contribution
SingleHeadSummary      — compact end-to-end single-head pipeline diagram

Usage
-----
    from attention.attention_output import WeightedSumAnim, AttentionOutputVector, SingleHeadSummary

    class MyScene(Scene):
        def construct(self):
            ws = WeightedSumAnim(
                weights=[0.6, 0.3, 0.1],
                value_rows=[[0.5, -0.3, 0.8], [0.2, 0.9, -0.1], [-0.4, 0.1, 0.5]],
            )
            self.play(ws.build_anim())
            self.wait()

Design Contract
---------------
Every public class:
  • Is a VGroup — safe to move / scale / transform like any Manim mob.
  • Exposes a `build_anim()` → AnimationGroup that plays the whole sequence.
  • Exposes fine-grained helpers (highlight, fade_out, etc.) for composition.
  • Reads colors from styles.colors  (falls back to inline defaults so the file
    works stand-alone before the full package exists).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from manim import (
    RIGHT, LEFT, UP, DOWN, ORIGIN,
    WHITE, BLACK, GRAY, YELLOW,
    AnimationGroup, FadeIn, FadeOut, GrowArrow,
    LaggedStart, Succession, Transform,
    always_redraw,
    VGroup, VMobject,
    Rectangle, RoundedRectangle, Arrow, Line,
    Text, MathTex, Tex,
    BarChart,
    DecimalNumber,
    Brace,
    Scene,
    Create, Write, Indicate,
    Dot,
    config,
    MoveAlongPath, CurvedArrow,
    rate_functions,
    smooth,
)
from manim import color as mcolor
from manim import interpolate_color, ManimColor

# ─────────────────────────────────────────────
# Attempt to import shared palette; fall back gracefully
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
    }

# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _hex(key: str) -> ManimColor:
    return ManimColor(COLORS.get(key, "#FFFFFF"))


def _lerp_color(c1: str, c2: str, t: float) -> ManimColor:
    return interpolate_color(ManimColor(c1), ManimColor(c2), t)


def _value_to_color(v: float, lo: float = -1.0, hi: float = 1.0) -> ManimColor:
    """Map a scalar value → color: negative=red, zero=gray, positive=blue."""
    t = (v - lo) / (hi - lo + 1e-9)
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return interpolate_color(ManimColor("#F44336"), ManimColor("#555555"), t * 2)
    else:
        return interpolate_color(ManimColor("#555555"), ManimColor("#2196F3"), (t - 0.5) * 2)


def _make_cell(
    value: float,
    width: float = 0.45,
    height: float = 0.45,
    show_val: bool = True,
    font_size: int = 14,
) -> VGroup:
    """Single matrix/vector cell — colored rectangle + optional value label."""
    rect = Rectangle(
        width=width, height=height,
        fill_color=_value_to_color(value),
        fill_opacity=0.85,
        stroke_color=WHITE,
        stroke_width=0.8,
    )
    cell = VGroup(rect)
    if show_val:
        lbl = Text(f"{value:+.2f}", font_size=font_size, color=WHITE)
        lbl.move_to(rect.get_center())
        cell.add(lbl)
    return cell


def _make_vector(
    values: Sequence[float],
    direction: str = "horizontal",
    cell_w: float = 0.45,
    cell_h: float = 0.45,
    show_val: bool = True,
    font_size: int = 14,
) -> VGroup:
    """Row or column of colored cells representing a vector."""
    cells = VGroup()
    for v in values:
        cells.add(_make_cell(v, cell_w, cell_h, show_val, font_size))
    if direction == "horizontal":
        cells.arrange(RIGHT, buff=0.04)
    else:
        cells.arrange(DOWN, buff=0.04)
    return cells


# ═══════════════════════════════════════════════════════════════════════════════
# Asset 1: WeightedSumAnim
# ═══════════════════════════════════════════════════════════════════════════════

class WeightedSumAnim(VGroup):
    """
    Animates: softmax_weights[i] × V[i]  →  sum  →  output vector.

    Layout (left to right)
    ──────────────────────
    Attention weights bar  |  Value matrix  |  Scaled rows  |  ∑  |  Output

    Parameters
    ----------
    weights     : 1-D list of floats that sum ≈ 1  (e.g. softmax outputs)
    value_rows  : list of equal-length float lists  (the V matrix rows)
    token_labels: optional token strings for row labels
    cell_size   : (width, height) of each matrix cell
    font_size   : label font size
    """

    def __init__(
        self,
        weights: Sequence[float],
        value_rows: Sequence[Sequence[float]],
        token_labels: Sequence[str] | None = None,
        cell_size: tuple[float, float] = (0.42, 0.42),
        font_size: int = 14,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert len(weights) == len(value_rows), "weights and value_rows must have the same length"
        self.weights = list(weights)
        self.value_rows = [list(r) for r in value_rows]
        self.n_tokens = len(weights)
        self.d_v = len(value_rows[0])
        self.token_labels = token_labels or [f"t{i}" for i in range(self.n_tokens)]
        self.cw, self.ch = cell_size
        self.fs = font_size

        # ── Build static sub-objects ──────────────────────────────────────────
        self._build_weight_bars()
        self._build_value_matrix()
        self._build_output_vector()
        self._build_labels()
        self._arrange_layout()

        # Register everything with VGroup so .move_to / .scale work
        self.add(
            self.weight_group,
            self.value_matrix_group,
            self.scaled_rows_group,
            self.sum_sym,
            self.output_group,
            self.label_group,
        )

    # ── Sub-object builders ───────────────────────────────────────────────────

    def _build_weight_bars(self):
        """Vertical weight bars, one per token."""
        bars = VGroup()
        max_h = self.ch * self.n_tokens * 0.85
        weight_colors = [
            _lerp_color("#9C27B0", "#FFEB3B", w)
            for w in self.weights
        ]
        self.weight_bars = []
        for i, (w, wc) in enumerate(zip(self.weights, weight_colors)):
            bar_h = max(w * max_h, 0.04)
            bar = Rectangle(
                width=self.cw * 1.1, height=bar_h,
                fill_color=wc, fill_opacity=0.9,
                stroke_color=WHITE, stroke_width=0.8,
            )
            pct = Text(f"{w:.2f}", font_size=self.fs, color=WHITE)
            pct.next_to(bar, RIGHT, buff=0.06)
            entry = VGroup(bar, pct)
            bars.add(entry)
            self.weight_bars.append(entry)
        bars.arrange(DOWN, buff=0.06)

        title = Text("Attn\nWeights", font_size=self.fs + 2, color=_hex("attention"))
        title.next_to(bars, UP, buff=0.15)
        self.weight_group = VGroup(title, bars)

    def _build_value_matrix(self):
        """V matrix — rows are Value vectors, shown as colored cells."""
        rows = VGroup()
        self.v_row_mobs = []
        for r in self.value_rows:
            row_mob = _make_vector(r, "horizontal", self.cw, self.ch, True, self.fs)
            rows.add(row_mob)
            self.v_row_mobs.append(row_mob)
        rows.arrange(DOWN, buff=0.06)

        title = Text("Value (V)", font_size=self.fs + 2, color=_hex("value"))
        title.next_to(rows, UP, buff=0.15)
        self.value_matrix_group = VGroup(title, rows)

    def _build_output_vector(self):
        """Output = weighted sum of V rows."""
        weights_arr = np.array(self.weights)
        v_arr = np.array(self.value_rows)
        self.output_vals = (weights_arr[:, None] * v_arr).sum(axis=0).tolist()

        # Scaled rows (w_i * V_i) — built but initially invisible; shown during anim
        self.scaled_row_mobs = []
        scaled_group = VGroup()
        for i, (w, row) in enumerate(zip(self.weights, self.value_rows)):
            scaled = [w * v for v in row]
            row_mob = _make_vector(scaled, "horizontal", self.cw, self.ch, True, self.fs)
            scaled_group.add(row_mob)
            self.scaled_row_mobs.append(row_mob)
        scaled_group.arrange(DOWN, buff=0.06)

        title = Text("w·V rows", font_size=self.fs + 2, color=_hex("highlight"))
        title.next_to(scaled_group, UP, buff=0.15)
        self.scaled_rows_group = VGroup(title, scaled_group)

        # Sum symbol
        self.sum_sym = MathTex(r"\sum", font_size=52, color=WHITE)

        # Output vector
        out_mob = _make_vector(self.output_vals, "horizontal", self.cw, self.ch, True, self.fs)
        out_title = Text("Output", font_size=self.fs + 2, color=_hex("output"))
        out_title.next_to(out_mob, UP, buff=0.15)
        self.output_group = VGroup(out_title, out_mob)
        self.out_vector_mob = out_mob

    def _build_labels(self):
        """Token row labels (left side of value matrix)."""
        labels = VGroup()
        for lbl, row_mob in zip(self.token_labels, self.v_row_mobs):
            t = Text(f'"{lbl}"', font_size=self.fs, color=_hex("token"))
            t.next_to(row_mob, LEFT, buff=0.15)
            labels.add(t)
        self.label_group = labels

    def _arrange_layout(self):
        """Position all groups left→right with spacing."""
        self.value_matrix_group.next_to(self.weight_group, RIGHT, buff=0.6)
        self.scaled_rows_group.next_to(self.value_matrix_group, RIGHT, buff=0.5)
        self.sum_sym.next_to(self.scaled_rows_group, RIGHT, buff=0.35)
        self.output_group.next_to(self.sum_sym, RIGHT, buff=0.35)

        # Align token labels with V rows after layout
        for lbl, row_mob in zip(self.label_group, self.v_row_mobs):
            lbl.next_to(row_mob, LEFT, buff=0.15)

    # ── Animation API ─────────────────────────────────────────────────────────

    def build_anim(self, run_time_per_step: float = 0.7) -> Succession:
        """
        Full animation sequence:
         1. Fade in weights + V matrix
         2. Row by row: highlight weight bar, scale V row → scaled row
         3. Collect scaled rows, show ∑ symbol
         4. Reveal output vector
        """
        rt = run_time_per_step
        steps = []

        # Step 1 — show weights & V
        steps.append(AnimationGroup(
            FadeIn(self.weight_group, shift=RIGHT * 0.2),
            FadeIn(self.value_matrix_group, shift=RIGHT * 0.2),
            FadeIn(self.label_group),
            lag_ratio=0.3,
            run_time=rt,
        ))

        # Step 2 — scaled row title appears
        title_mob = self.scaled_rows_group[0]
        steps.append(FadeIn(title_mob, run_time=rt * 0.5))

        # Step 3 — row by row scaling
        for i in range(self.n_tokens):
            bar_entry = self.weight_bars[i]
            v_row = self.v_row_mobs[i]
            scaled_row = self.scaled_row_mobs[i]

            steps.append(AnimationGroup(
                Indicate(bar_entry, color=_hex("highlight"), scale_factor=1.15, run_time=rt * 0.6),
                Indicate(v_row, color=_hex("value"), scale_factor=1.1, run_time=rt * 0.6),
                lag_ratio=0.2,
            ))
            steps.append(FadeIn(scaled_row, shift=RIGHT * 0.15, run_time=rt * 0.5))

        # Step 4 — sum symbol + output
        steps.append(AnimationGroup(
            Write(self.sum_sym, run_time=rt),
            FadeIn(self.output_group, shift=RIGHT * 0.2, run_time=rt),
            lag_ratio=0.4,
        ))

        # Step 5 — pulse the output
        steps.append(Indicate(
            self.out_vector_mob,
            color=_hex("output"),
            scale_factor=1.08,
            run_time=rt,
        ))

        return Succession(*steps)

    def highlight_output(self) -> Indicate:
        """Return an Indicate animation on the output vector."""
        return Indicate(self.out_vector_mob, color=_hex("highlight"), scale_factor=1.1)

    def fade_out_internals(self) -> AnimationGroup:
        """Fade everything except the output vector."""
        return AnimationGroup(
            FadeOut(self.weight_group),
            FadeOut(self.value_matrix_group),
            FadeOut(self.scaled_rows_group),
            FadeOut(self.sum_sym),
            FadeOut(self.label_group),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Asset 2: AttentionOutputVector
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionOutputVector(VGroup):
    """
    Displays the final attention output vector alongside a contribution
    bar chart showing which input tokens drove the output most.

    Parameters
    ----------
    output_values   : the output vector (d_v floats)
    weights         : attention weights (n_tokens floats)
    token_labels    : token strings for the contribution bar chart
    cell_size       : (w, h) per vector cell
    bar_max_width   : max width of contribution bars
    """

    def __init__(
        self,
        output_values: Sequence[float],
        weights: Sequence[float],
        token_labels: Sequence[str] | None = None,
        cell_size: tuple[float, float] = (0.42, 0.42),
        bar_max_width: float = 1.6,
        font_size: int = 14,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_values = list(output_values)
        self.weights = list(weights)
        self.n_tokens = len(weights)
        self.token_labels = token_labels or [f"t{i}" for i in range(self.n_tokens)]
        self.cw, self.ch = cell_size
        self.bar_max_width = bar_max_width
        self.fs = font_size

        self._build_vector_display()
        self._build_contribution_bars()
        self._arrange()
        self.add(self.vector_group, self.contrib_group)

    def _build_vector_display(self):
        vec = _make_vector(self.output_values, "horizontal", self.cw, self.ch, True, self.fs)
        title = Text("Attention Output", font_size=self.fs + 3, color=_hex("output"))
        title.next_to(vec, UP, buff=0.12)

        d_label = Text(f"dim = {len(self.output_values)}", font_size=self.fs - 1, color=GRAY)
        d_label.next_to(vec, DOWN, buff=0.1)

        self.vec_mob = vec
        self.vector_group = VGroup(title, vec, d_label)

    def _build_contribution_bars(self):
        """Horizontal bar chart — bar width ∝ attention weight."""
        bars = VGroup()
        self.contrib_bars = []
        for i, (w, lbl) in enumerate(zip(self.weights, self.token_labels)):
            bar_w = max(w * self.bar_max_width, 0.04)
            color = _lerp_color("#9C27B0", "#FFEB3B", w)
            bar = Rectangle(
                width=bar_w, height=self.ch * 0.75,
                fill_color=color, fill_opacity=0.9,
                stroke_color=WHITE, stroke_width=0.6,
            )
            tok_lbl = Text(f'"{lbl}"', font_size=self.fs - 1, color=_hex("token"))
            tok_lbl.next_to(bar, LEFT, buff=0.1)
            pct_lbl = Text(f"{w:.0%}", font_size=self.fs - 1, color=WHITE)
            pct_lbl.next_to(bar, RIGHT, buff=0.08)
            row = VGroup(tok_lbl, bar, pct_lbl)
            bars.add(row)
            self.contrib_bars.append((bar, tok_lbl, pct_lbl))
        bars.arrange(DOWN, buff=0.12)

        title = Text("Token Contributions", font_size=self.fs + 2, color=_hex("attention"))
        title.next_to(bars, UP, buff=0.15)
        self.contrib_group = VGroup(title, bars)

    def _arrange(self):
        self.contrib_group.next_to(self.vector_group, RIGHT, buff=0.55)

    # ── Animation API ─────────────────────────────────────────────────────────

    def build_anim(self, run_time: float = 0.6) -> Succession:
        """
        1. Reveal vector cells left → right
        2. Grow contribution bars top → bottom (scaled by weight)
        """
        steps = []

        # 1. Vector reveal
        steps.append(LaggedStart(
            *[FadeIn(cell, shift=UP * 0.1) for cell in self.vec_mob],
            lag_ratio=0.15,
            run_time=run_time * self.cw * len(self.output_values),
        ))

        # 2. Contribution bars
        contrib_anims = []
        for bar, tok_lbl, pct_lbl in self.contrib_bars:
            contrib_anims.append(AnimationGroup(
                FadeIn(tok_lbl),
                Create(bar),
                FadeIn(pct_lbl),
                lag_ratio=0.2,
            ))
        steps.append(LaggedStart(*contrib_anims, lag_ratio=0.25, run_time=run_time * 2))

        return Succession(*steps)

    def highlight_top_contributor(self) -> AnimationGroup:
        """Indicate the bar with the highest attention weight."""
        top_idx = int(np.argmax(self.weights))
        bar, tok_lbl, _ = self.contrib_bars[top_idx]
        return AnimationGroup(
            Indicate(bar, color=_hex("highlight"), scale_factor=1.12),
            Indicate(tok_lbl, color=_hex("highlight"), scale_factor=1.1),
        )

    def connect_to(self, other: VMobject) -> CurvedArrow:
        """Return a CurvedArrow from this output to another mobject."""
        return CurvedArrow(
            self.vec_mob.get_right() + RIGHT * 0.05,
            other.get_left() + LEFT * 0.05,
            angle=-0.3,
            color=_hex("arrow"),
            stroke_width=2,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Asset 3: SingleHeadSummary
# ═══════════════════════════════════════════════════════════════════════════════

class SingleHeadSummary(VGroup):
    """
    Compact end-to-end diagram of a single attention head.

    Layout (left → right):
      Input  →  Q K V  →  Scores  →  Softmax  →  Weights  →  × V  →  Output

    Each stage is a labeled box; arrows connect them.
    Stages light up sequentially via build_anim().

    Parameters
    ----------
    seq_len  : number of tokens
    d_k      : key/query dimension
    d_v      : value dimension (= output dimension per head)
    compact  : if True, boxes are smaller (fits on screen alongside other content)
    """

    # Stage definitions: (id, label, color_key)
    _STAGES = [
        ("input",    "Input\nX",       "embedding"),
        ("proj",     "Q  K  V\nproj.", "attention"),
        ("scores",   "QKᵀ/√dₖ\nScores","query"),
        ("softmax",  "Softmax\n→ Weights","key"),
        ("weighted", "× V\nWeighted Sum","value"),
        ("output",   "Head\nOutput",   "output"),
    ]

    def __init__(
        self,
        seq_len: int = 4,
        d_k: int = 64,
        d_v: int = 64,
        compact: bool = False,
        font_size: int = 15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_k = d_k
        self.d_v = d_v
        self.fs = font_size - (3 if compact else 0)
        self.box_w = 1.1 if compact else 1.4
        self.box_h = 0.75 if compact else 0.9

        self.stage_boxes: dict[str, VGroup] = {}
        self.arrows: list[Arrow] = []

        self._build_stages()
        self._build_arrows()
        self._build_dim_labels(compact)
        self.add(*self.stage_boxes.values(), *self.arrows, self.dim_label_group)

    def _make_stage_box(self, label: str, color_key: str) -> VGroup:
        rect = RoundedRectangle(
            width=self.box_w, height=self.box_h,
            corner_radius=0.12,
            fill_color=_hex(color_key),
            fill_opacity=0.25,
            stroke_color=_hex(color_key),
            stroke_width=2,
        )
        lbl = Text(label, font_size=self.fs, color=WHITE)
        lbl.move_to(rect.get_center())
        return VGroup(rect, lbl)

    def _build_stages(self):
        boxes = VGroup()
        for sid, label, color_key in self._STAGES:
            box = self._make_stage_box(label, color_key)
            self.stage_boxes[sid] = box
            boxes.add(box)
        boxes.arrange(RIGHT, buff=0.28)

    def _build_arrows(self):
        stage_ids = [s[0] for s in self._STAGES]
        for i in range(len(stage_ids) - 1):
            src = self.stage_boxes[stage_ids[i]]
            tgt = self.stage_boxes[stage_ids[i + 1]]
            arr = Arrow(
                src.get_right(), tgt.get_left(),
                buff=0.06,
                color=_hex("arrow"),
                stroke_width=2,
                max_tip_length_to_length_ratio=0.25,
            )
            self.arrows.append(arr)

    def _build_dim_labels(self, compact: bool):
        """Small dimension annotations under each arrow."""
        dim_labels = {
            0: f"{self.seq_len}×d",
            1: f"{self.seq_len}×{self.d_k}",
            2: f"{self.seq_len}²",
            3: f"{self.seq_len}×{self.seq_len}",
            4: f"{self.seq_len}×{self.d_v}",
        }
        grp = VGroup()
        for i, arr in enumerate(self.arrows):
            if i in dim_labels:
                lbl = Text(dim_labels[i], font_size=self.fs - 3, color=GRAY)
                lbl.next_to(arr, DOWN, buff=0.06)
                grp.add(lbl)
        self.dim_label_group = grp

    # ── Animation API ─────────────────────────────────────────────────────────

    def build_anim(self, run_time_per_stage: float = 0.5) -> Succession:
        """
        Light up each stage box and its connecting arrow in sequence.
        """
        rt = run_time_per_stage
        stage_ids = [s[0] for s in self._STAGES]
        steps = []

        for i, sid in enumerate(stage_ids):
            box = self.stage_boxes[sid]
            anims = [FadeIn(box, scale=0.85, run_time=rt)]
            if i < len(self.arrows):
                anims.append(GrowArrow(self.arrows[i], run_time=rt * 0.6))
                if i < len(self.dim_label_group):
                    anims.append(FadeIn(self.dim_label_group[i], run_time=rt * 0.5))
            steps.append(AnimationGroup(*anims, lag_ratio=0.3))

        return Succession(*steps)

    def highlight_stage(self, stage_id: str) -> Indicate:
        """Highlight a specific stage by its id string."""
        box = self.stage_boxes.get(stage_id)
        if box is None:
            raise ValueError(f"Unknown stage '{stage_id}'. Valid: {list(self.stage_boxes)}")
        return Indicate(box, color=_hex("highlight"), scale_factor=1.12)

    def expand_stage(self, stage_id: str, detail_mob: VMobject, scene: Scene) -> None:
        """
        Helper: zoom into a stage box and overlay a detail mobject.
        Call from scene.construct() — not an animation object itself.

        Example
        -------
            summary.expand_stage("softmax", my_softmax_diagram, self)
        """
        box = self.stage_boxes[stage_id]
        detail_mob.move_to(box.get_center())
        scene.play(
            self.animate.scale(0.5).to_corner(LEFT + DOWN),
            FadeIn(detail_mob),
            run_time=0.8,
        )

    def connect_to(self, other: VMobject, from_stage: str = "output") -> Arrow:
        """Arrow from a stage's right edge to another mobject."""
        src_box = self.stage_boxes[from_stage]
        return Arrow(
            src_box.get_right(),
            other.get_left(),
            buff=0.1,
            color=_hex("arrow"),
            stroke_width=2,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Quick smoke-test scene (run with: manim -pql attention_output.py DemoScene)
# ═══════════════════════════════════════════════════════════════════════════════

class DemoScene(Scene):
    """Demonstrates all three assets together."""

    def construct(self):
        # ── 1. WeightedSumAnim ────────────────────────────────────────────────
        ws = WeightedSumAnim(
            weights=[0.60, 0.30, 0.10],
            value_rows=[
                [ 0.5, -0.3,  0.8],
                [ 0.2,  0.9, -0.1],
                [-0.4,  0.1,  0.5],
            ],
            token_labels=["cat", "sat", "on"],
        )
        ws.scale(0.85).move_to(ORIGIN)
        self.play(ws.build_anim(run_time_per_step=0.55))
        self.play(ws.highlight_output())
        self.wait(0.5)
        self.play(FadeOut(ws))

        # ── 2. AttentionOutputVector ──────────────────────────────────────────
        weights = [0.60, 0.30, 0.10]
        v_rows  = [[ 0.5, -0.3,  0.8], [ 0.2,  0.9, -0.1], [-0.4,  0.1,  0.5]]
        w = np.array(weights); V = np.array(v_rows)
        output_vals = (w[:, None] * V).sum(axis=0).tolist()

        aov = AttentionOutputVector(
            output_values=output_vals,
            weights=weights,
            token_labels=["cat", "sat", "on"],
        )
        aov.scale(0.9).move_to(ORIGIN)
        self.play(aov.build_anim(run_time=0.5))
        self.play(aov.highlight_top_contributor())
        self.wait(0.5)
        self.play(FadeOut(aov))

        # ── 3. SingleHeadSummary ──────────────────────────────────────────────
        summary = SingleHeadSummary(seq_len=4, d_k=64, d_v=64)
        summary.scale(0.75).move_to(ORIGIN)
        self.play(summary.build_anim(run_time_per_stage=0.45))
        self.play(summary.highlight_stage("softmax"))
        self.play(summary.highlight_stage("weighted"))
        self.wait(1)