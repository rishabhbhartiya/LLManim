"""
attention/multi_head.py
───────────────────────
Reusable Manim assets for Multi-Head Attention (MHA).

  d_model input  →  split into h heads  →  each head runs independently
  →  concatenate h outputs  →  W_O projection  →  d_model output

Assets
------
HeadSplitAnim           — d_model vector splits into h equal chunks, one per head
MultiHeadGrid           — h attention heads running in parallel with heatmaps
HeadConcatAnim          — h output vectors concatenate into one long vector
OutputProjectionAnim    — concatenated vector × W_O → back to d_model
HeadComparisonDisplay   — all heads' heatmaps tiled with pattern labels

Usage
-----
    from attention.multi_head import (
        HeadSplitAnim, MultiHeadGrid,
        HeadConcatAnim, OutputProjectionAnim,
        HeadComparisonDisplay,
    )

Design contract
---------------
  • Every class is a VGroup — freely composable.
  • build_anim() → Succession plays the full sequence.
  • Fine-grained helpers for scene-level composition.
  • Graceful fallback if styles/colors.py is absent.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from manim import (
    RIGHT, LEFT, UP, DOWN, ORIGIN, UL, UR, DL, DR,
    WHITE, BLACK, GRAY, YELLOW,
    AnimationGroup, FadeIn, FadeOut, GrowArrow,
    LaggedStart, Succession, Transform,
    VGroup, VMobject,
    Rectangle, RoundedRectangle, Square,
    Arrow, CurvedArrow, Line, DashedLine,
    Text, MathTex,
    Brace,
    Scene,
    Create, Write, Indicate, Flash,
    SurroundingRectangle,
    DEGREES,
    config,
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
    }

def _hex(key: str) -> ManimColor:
    return ManimColor(COLORS.get(key, "#FFFFFF"))

def _lerp(c1: str, c2: str, t: float) -> ManimColor:
    return interpolate_color(ManimColor(c1), ManimColor(c2), float(np.clip(t, 0, 1)))

# ── Deterministic per-head color palette ─────────────────────────────────────
# 8 visually distinct hues cycling for any number of heads
_HEAD_PALETTE = [
    "#E91E63",  # pink   (query-like)
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#F44336",  # red
    "#FFEB3B",  # yellow
]

def _head_color(h_idx: int) -> ManimColor:
    return ManimColor(_HEAD_PALETTE[h_idx % len(_HEAD_PALETTE)])

def _prob_color(p: float, h_idx: int = 0) -> ManimColor:
    """Per-head heatmap: dim → head's accent color."""
    return _lerp(COLORS["dim"], _HEAD_PALETTE[h_idx % len(_HEAD_PALETTE)], p)


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _seg_rect(
    width: float, height: float,
    color: ManimColor,
    label: str = "",
    font_size: int = 12,
    opacity: float = 0.85,
) -> VGroup:
    """A colored rectangle segment with an optional centered label."""
    rect = Rectangle(
        width=width, height=height,
        fill_color=color,
        fill_opacity=opacity,
        stroke_color=WHITE,
        stroke_width=0.8,
    )
    grp = VGroup(rect)
    if label:
        lbl = Text(label, font_size=font_size, color=WHITE)
        lbl.move_to(rect.get_center())
        grp.add(lbl)
    return grp


def _softmax(scores: np.ndarray) -> np.ndarray:
    s = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def _make_random_weights(seq_len: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-2, 2, (seq_len, seq_len))
    return _softmax(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# Asset 1 — HeadSplitAnim
# ═══════════════════════════════════════════════════════════════════════════════

class HeadSplitAnim(VGroup):
    """
    Animates a d_model vector splitting into h equal sub-vectors (heads).

    Layout
    ──────
                   ┌─ Head 0 (d_k dims) ──┐
    [  d_model  ]──┼─ Head 1 (d_k dims) ──┤
                   └─ Head h-1 (d_k) ─────┘

    Each chunk gets its own color. Connecting arrows fan out from the
    source vector to each head chunk.

    Parameters
    ----------
    d_model     : total embedding dimension
    num_heads   : number of attention heads  (d_model must be divisible)
    token_label : optional label for the source vector (e.g. "x[cat]")
    cell_height : height of each cell segment
    vec_width   : total width of the source vector bar
    font_size   : text size
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        token_label: str = "x",
        cell_height: float = 0.55,
        vec_width: float = 3.6,
        font_size: int = 13,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.token_label = token_label
        self.cell_h = cell_height
        self.vec_w = vec_width
        self.fs = font_size

        self._build()
        self.add(
            self.source_group,
            self.head_chunks,
            self.fan_arrows,
            self.dim_labels,
        )

    def _build(self):
        # ── Source vector ─────────────────────────────────────────────────────
        src_rect = Rectangle(
            width=self.vec_w, height=self.cell_h,
            fill_color=_hex("embedding"),
            fill_opacity=0.8,
            stroke_color=WHITE, stroke_width=1.2,
        )
        src_lbl = Text(
            f"{self.token_label}  [{self.d_model}]",
            font_size=self.fs + 1, color=WHITE,
        )
        src_lbl.move_to(src_rect.get_center())
        src_title = Text("Input vector", font_size=self.fs, color=_hex("embedding"))
        src_title.next_to(src_rect, UP, buff=0.12)
        self.source_group = VGroup(src_title, src_rect)
        self.src_rect = src_rect

        # ── Head chunks ───────────────────────────────────────────────────────
        chunk_w = self.vec_w / self.num_heads
        chunks = VGroup()
        self.chunk_rects = []
        for h in range(self.num_heads):
            color = _head_color(h)
            seg = _seg_rect(
                chunk_w - 0.04, self.cell_h,
                color,
                label=f"h{h}" if chunk_w > 0.5 else "",
                font_size=self.fs - 2,
            )
            chunks.add(seg)
            self.chunk_rects.append(seg)
        chunks.arrange(RIGHT, buff=0.04)

        # Head output labels below
        head_labels = VGroup()
        for h in range(self.num_heads):
            lbl = Text(f"Head {h}", font_size=self.fs - 1,
                       color=_head_color(h))
            lbl.next_to(self.chunk_rects[h], DOWN, buff=0.1)
            head_labels.add(lbl)
        self.head_labels = head_labels

        self.head_chunks = VGroup(chunks, head_labels)
        self.head_chunks.next_to(self.source_group, DOWN, buff=0.9)

        # ── Fan-out arrows ────────────────────────────────────────────────────
        self.fan_arrows = VGroup()
        src_bottom = self.src_rect.get_bottom()
        for h, seg in enumerate(self.chunk_rects):
            arr = Arrow(
                src_bottom,
                seg.get_top(),
                buff=0.08,
                color=_head_color(h),
                stroke_width=1.6,
                max_tip_length_to_length_ratio=0.18,
            )
            self.fan_arrows.add(arr)

        # ── Dimension brace ───────────────────────────────────────────────────
        brace = Brace(self.chunk_rects[0], DOWN, buff=0.5, color=GRAY)
        brace_lbl = brace.get_text(
            f"d_k = {self.d_k}", buff=0.08,
        )
        brace_lbl.set(font_size=self.fs - 1, color=GRAY)
        full_brace = Brace(chunks, UP, buff=0.04, color=GRAY)
        full_brace_lbl = full_brace.get_text(
            f"d_model = {self.d_model}", buff=0.08,
        )
        full_brace_lbl.set(font_size=self.fs - 1, color=GRAY)
        self.dim_labels = VGroup(brace, brace_lbl, full_brace, full_brace_lbl)

    # ── Animation API ─────────────────────────────────────────────────────────

    def build_anim(self, run_time: float = 0.55) -> Succession:
        steps = []

        # 1. Source vector
        steps.append(FadeIn(self.source_group, shift=DOWN * 0.1, run_time=run_time))

        # 2. Fan-out arrows simultaneously
        steps.append(LaggedStart(
            *[GrowArrow(a) for a in self.fan_arrows],
            lag_ratio=0.08, run_time=run_time,
        ))

        # 3. Head chunks appear with color pop
        steps.append(LaggedStart(
            *[FadeIn(seg, scale=0.7) for seg in self.chunk_rects],
            lag_ratio=0.08, run_time=run_time,
        ))
        steps.append(LaggedStart(
            *[FadeIn(lbl, shift=DOWN * 0.08) for lbl in self.head_labels],
            lag_ratio=0.08, run_time=run_time * 0.6,
        ))

        # 4. Dimension braces
        steps.append(FadeIn(self.dim_labels, run_time=run_time * 0.7))

        return Succession(*steps)

    def highlight_head(self, h_idx: int) -> AnimationGroup:
        return AnimationGroup(
            Indicate(self.chunk_rects[h_idx],
                     color=_hex("highlight"), scale_factor=1.15),
            Indicate(self.head_labels[h_idx],
                     color=_hex("highlight"), scale_factor=1.1),
        )

    def get_head_position(self, h_idx: int):
        """Return the center point of head h's chunk (for arrow targets)."""
        return self.chunk_rects[h_idx].get_center()


# ═══════════════════════════════════════════════════════════════════════════════
# Asset 2 — MultiHeadGrid
# ═══════════════════════════════════════════════════════════════════════════════

class MultiHeadGrid(VGroup):
    """
    Displays h attention heads running in parallel, each with its own
    compact heatmap and a "Head N" label.

    Each head can be synchronized (played together) or stepped individually.

    Parameters
    ----------
    num_heads       : number of heads to show (≤ 8 looks good on screen)
    seq_len         : sequence length (heatmap size)
    token_labels    : token strings for axis labels
    weights_list    : optional list of (seq_len × seq_len) weight arrays,
                      one per head. If None, random weights are generated.
    cell_size       : (w, h) per heatmap cell
    heads_per_row   : how many heads to place in one row (auto if None)
    font_size       : text size
    """

    def __init__(
        self,
        num_heads: int = 4,
        seq_len: int = 4,
        token_labels: Sequence[str] | None = None,
        weights_list: Sequence[Sequence[Sequence[float]]] | None = None,
        cell_size: tuple[float, float] = (0.30, 0.30),
        heads_per_row: int | None = None,
        font_size: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.token_labels = token_labels or [f"t{i}" for i in range(seq_len)]
        self.cw, self.ch = cell_size
        self.fs = font_size

        if weights_list is not None:
            self.weights = [np.array(w) for w in weights_list]
        else:
            self.weights = [
                _make_random_weights(seq_len, seed=h)
                for h in range(num_heads)
            ]

        self._hpr = heads_per_row or min(num_heads, 4)
        self._build()
        self.add(self.grid_group)

    def _single_heatmap(self, h_idx: int) -> tuple[VGroup, list[list[VGroup]]]:
        """Build one head's heatmap. Returns (full_group, cell_mobs)."""
        w_mat = self.weights[h_idx]
        hcolor = _head_color(h_idx)

        # Title
        title = Text(f"Head {h_idx}", font_size=self.fs + 2, color=hcolor)

        # Cells
        cell_mobs: list[list[VGroup]] = []
        grid = VGroup()
        for i in range(self.seq_len):
            row_grp = VGroup()
            row_cells = []
            for j in range(self.seq_len):
                p = float(w_mat[i, j])
                rect = Rectangle(
                    width=self.cw, height=self.ch,
                    fill_color=_prob_color(p, h_idx),
                    fill_opacity=0.90,
                    stroke_color=WHITE, stroke_width=0.4,
                )
                cell = VGroup(rect)
                row_grp.add(cell)
                row_cells.append(cell)
            row_grp.arrange(RIGHT, buff=0.02)
            grid.add(row_grp)
            cell_mobs.append(row_cells)
        grid.arrange(DOWN, buff=0.02)

        # Thin border around the heatmap
        border = SurroundingRectangle(
            grid, color=hcolor, buff=0.06, stroke_width=1.5,
        )

        title.next_to(grid, UP, buff=0.1)
        head_group = VGroup(title, grid, border)
        return head_group, cell_mobs

    def _build(self):
        self.head_groups: list[VGroup] = []
        self.head_cells: list[list[list[VGroup]]] = []

        all_heads = VGroup()
        for h in range(self.num_heads):
            hg, cells = self._single_heatmap(h)
            self.head_groups.append(hg)
            self.head_cells.append(cells)
            all_heads.add(hg)

        # Arrange in rows
        rows = VGroup()
        hpr = self._hpr
        for start in range(0, self.num_heads, hpr):
            row_group = VGroup(*all_heads[start: start + hpr])
            row_group.arrange(RIGHT, buff=0.35)
            rows.add(row_group)
        rows.arrange(DOWN, buff=0.45)
        self.grid_group = rows

    # ── Animation API ─────────────────────────────────────────────────────────

    def build_anim(self, run_time_per_head: float = 0.45) -> Succession:
        """
        Reveal all heads simultaneously, cell by cell with a lag.
        Title + border appear first, then cells sweep in.
        """
        steps = []

        # Titles + borders
        title_border_anims = []
        for hg in self.head_groups:
            title_border_anims.append(FadeIn(hg[0]))  # title
            title_border_anims.append(Create(hg[2]))   # border
        steps.append(LaggedStart(*title_border_anims, lag_ratio=0.1,
                                  run_time=run_time_per_head))

        # Cells — all heads in parallel, each row lagged
        cell_anims = []
        for h, cell_rows in enumerate(self.head_cells):
            for i, row_cells in enumerate(cell_rows):
                for j, cell in enumerate(row_cells):
                    cell_anims.append(
                        FadeIn(cell, scale=0.5,
                               run_time=run_time_per_head * 0.5)
                    )
        steps.append(LaggedStart(*cell_anims, lag_ratio=0.03,
                                  run_time=run_time_per_head * self.seq_len))

        return Succession(*steps)

    def play_head(self, h_idx: int) -> LaggedStart:
        """Return animation that reveals only head h's cells."""
        cells = self.head_cells[h_idx]
        anims = [FadeIn(c, scale=0.5) for row in cells for c in row]
        return LaggedStart(*anims, lag_ratio=0.05)

    def highlight_head(self, h_idx: int) -> AnimationGroup:
        hg = self.head_groups[h_idx]
        return AnimationGroup(
            Indicate(hg[2], color=_hex("highlight"), scale_factor=1.08),  # border
            Indicate(hg[0], color=_hex("highlight"), scale_factor=1.1),   # title
        )

    def highlight_cell(self, h_idx: int, i: int, j: int) -> Indicate:
        return Indicate(
            self.head_cells[h_idx][i][j],
            color=_hex("highlight"), scale_factor=1.3,
        )

    def get_head_group(self, h_idx: int) -> VGroup:
        return self.head_groups[h_idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Asset 3 — HeadConcatAnim
# ═══════════════════════════════════════════════════════════════════════════════

class HeadConcatAnim(VGroup):
    """
    Animates h head output vectors concatenating into one long vector.

    Layout (left → right):
      [head0 | head1 | … | head_{h-1}]  →  concat arrow  →  [full concat vec]

    Each head's sub-vector has its own color; the concatenated result
    shows all colors side by side.

    Parameters
    ----------
    num_heads       : number of heads
    d_v             : value/output dimension per head
    seq_token_idx   : which position's output to visualize (just a label)
    vec_cell_w      : width of each dim cell
    vec_cell_h      : height of cells
    font_size       : text size
    """

    def __init__(
        self,
        num_heads: int = 4,
        d_v: int = 64,
        seq_token_idx: int = 0,
        vec_cell_w: float = 0.13,
        vec_cell_h: float = 0.55,
        font_size: int = 13,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_v = d_v
        self.token_idx = seq_token_idx
        self.cw = vec_cell_w
        self.ch = vec_cell_h
        self.fs = font_size
        self.d_model = num_heads * d_v

        self._build()
        self.add(
            self.head_vecs_group,
            self.concat_arrow,
            self.concat_vec_group,
            self.dim_brace_group,
        )

    def _head_vec_mob(self, h_idx: int) -> VGroup:
        """Single head's output vector — d_v colored segments."""
        color = _head_color(h_idx)
        segs = VGroup()
        n_shown = min(self.d_v, 12)   # cap for readability
        for k in range(n_shown):
            seg = Rectangle(
                width=self.cw, height=self.ch,
                fill_color=color,
                fill_opacity=0.5 + 0.4 * (k / max(n_shown - 1, 1)),
                stroke_color=WHITE, stroke_width=0.5,
            )
            segs.add(seg)
        if self.d_v > n_shown:
            dots = Text("…", font_size=self.fs - 2, color=color)
            segs.add(dots)
        segs.arrange(RIGHT, buff=0.02)

        lbl = Text(f"head_{h_idx}", font_size=self.fs - 1, color=color)
        lbl.next_to(segs, UP, buff=0.1)
        dim_lbl = Text(f"[{self.d_v}]", font_size=self.fs - 3, color=GRAY)
        dim_lbl.next_to(segs, DOWN, buff=0.06)
        return VGroup(lbl, segs, dim_lbl)

    def _build(self):
        # Individual head output vectors
        head_vecs = VGroup()
        self.head_vec_mobs = []
        for h in range(self.num_heads):
            hv = self._head_vec_mob(h)
            head_vecs.add(hv)
            self.head_vec_mobs.append(hv)
        head_vecs.arrange(RIGHT, buff=0.22)

        title = Text(f"Head outputs  (token {self.token_idx})",
                     font_size=self.fs + 1, color=GRAY)
        title.next_to(head_vecs, UP, buff=0.2)
        self.head_vecs_group = VGroup(title, head_vecs)

        # Concat arrow
        self.concat_arrow = Arrow(
            LEFT * 0.1, RIGHT * 0.1,
            color=WHITE, stroke_width=2.5,
            max_tip_length_to_length_ratio=0.25,
        )
        concat_lbl = Text("concat", font_size=self.fs - 1, color=GRAY)
        concat_lbl.next_to(self.concat_arrow, UP, buff=0.06)
        self.concat_arrow = VGroup(self.concat_arrow, concat_lbl)

        # Concatenated result vector
        n_shown = min(self.d_model, 24)
        concat_segs = VGroup()
        for h in range(self.num_heads):
            color = _head_color(h)
            n_h = min(self.d_v, n_shown // self.num_heads + 1)
            for k in range(n_h):
                seg = Rectangle(
                    width=self.cw * 0.85, height=self.ch,
                    fill_color=color,
                    fill_opacity=0.5 + 0.4 * (k / max(n_h - 1, 1)),
                    stroke_color=WHITE, stroke_width=0.4,
                )
                concat_segs.add(seg)
        concat_segs.arrange(RIGHT, buff=0.015)
        self.concat_segs = concat_segs

        concat_title = Text(
            f"Concatenated  [{self.d_model}]",
            font_size=self.fs + 1, color=WHITE,
        )
        concat_title.next_to(concat_segs, UP, buff=0.12)
        self.concat_vec_group = VGroup(concat_title, concat_segs)

        # Layout
        self.head_vecs_group.move_to(ORIGIN)
        self.concat_arrow.next_to(self.head_vecs_group, DOWN, buff=0.4)
        self.concat_vec_group.next_to(self.concat_arrow, DOWN, buff=0.35)

        # Dimension brace
        brace = Brace(concat_segs, DOWN, buff=0.1, color=GRAY)
        brace_lbl = brace.get_text(
            f"{self.num_heads} × {self.d_v} = {self.d_model} dims",
        )
        brace_lbl.set(font_size=self.fs - 1, color=GRAY)
        self.dim_brace_group = VGroup(brace, brace_lbl)

    # ── Animation API ─────────────────────────────────────────────────────────

    def build_anim(self, run_time: float = 0.55) -> Succession:
        steps = []

        # 1. Head vectors appear
        steps.append(AnimationGroup(
            FadeIn(self.head_vecs_group[0]),  # title
            LaggedStart(
                *[FadeIn(hv, shift=DOWN * 0.1) for hv in self.head_vec_mobs],
                lag_ratio=0.15,
            ),
            lag_ratio=0.2, run_time=run_time,
        ))

        # 2. Concat arrow
        steps.append(FadeIn(self.concat_arrow, shift=DOWN * 0.05,
                             run_time=run_time * 0.5))

        # 3. Concat result — segments stream in left to right
        steps.append(AnimationGroup(
            FadeIn(self.concat_vec_group[0]),  # title
            LaggedStart(
                *[FadeIn(seg, scale=0.6) for seg in self.concat_segs],
                lag_ratio=0.025,
            ),
            lag_ratio=0.2, run_time=run_time * 1.5,
        ))

        # 4. Brace
        steps.append(FadeIn(self.dim_brace_group, run_time=run_time * 0.6))

        return Succession(*steps)

    def highlight_head_slice(self, h_idx: int) -> Indicate:
        return Indicate(
            self.head_vec_mobs[h_idx],
            color=_head_color(h_idx),
            scale_factor=1.1,
        )

    def get_concat_vec(self) -> VGroup:
        return self.concat_segs


# ═══════════════════════════════════════════════════════════════════════════════
# Asset 4 — OutputProjectionAnim
# ═══════════════════════════════════════════════════════════════════════════════

class OutputProjectionAnim(VGroup):
    """
    Concatenated vector  ×  W_O  →  d_model output.

    Layout (left → right):
      [concat, d_model]  ×  [W_O, d_model × d_model]  =  [output, d_model]

    The multiplication is shown as an animated column-highlight flow.

    Parameters
    ----------
    d_model     : full model dimension
    num_heads   : number of heads (for color coding the input)
    cell_size   : (w, h) of each matrix cell
    font_size   : text size
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 4,
        cell_size: tuple[float, float] = (0.18, 0.18),
        font_size: int = 12,
        n_display: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.cw, self.ch = cell_size
        self.fs = font_size
        self.nd = n_display   # displayed dims (full d_model is too wide)
        self.d_k = d_model // num_heads

        self._build()
        self.add(
            self.input_group,
            self.times_sym,
            self.wo_group,
            self.eq_sym,
            self.output_group,
            self.label_row,
        )

    def _color_block(self, rows: int, cols: int, color_fn) -> VGroup:
        """Grid of colored cells."""
        grid = VGroup()
        for i in range(rows):
            row_grp = VGroup()
            for j in range(cols):
                rect = Rectangle(
                    width=self.cw, height=self.ch,
                    fill_color=color_fn(i, j),
                    fill_opacity=0.82,
                    stroke_color=WHITE, stroke_width=0.3,
                )
                row_grp.add(rect)
            row_grp.arrange(RIGHT, buff=0.02)
            grid.add(row_grp)
        grid.arrange(DOWN, buff=0.02)
        return grid

    def _build(self):
        nd = self.nd
        n_h = self.num_heads
        d_k_shown = max(nd // n_h, 1)

        # Input vector (column vector, nd rows × 1)
        def inp_color(i, j):
            h = (i * n_h) // nd
            return _head_color(h)

        inp_grid = self._color_block(nd, 1, inp_color)
        inp_title = Text(f"Concat\n[{self.d_model}]", font_size=self.fs,
                         color=WHITE)
        inp_title.next_to(inp_grid, UP, buff=0.12)
        self.input_group = VGroup(inp_title, inp_grid)
        self.inp_grid = inp_grid

        # W_O matrix (nd × nd)
        def wo_color(i, j):
            return _lerp(COLORS["dim"], COLORS["weight_matrix"],
                         float(np.random.default_rng(i * nd + j).uniform(0.3, 1.0)))

        wo_grid = self._color_block(nd, nd, wo_color)
        wo_title = Text(f"W_O\n[{self.d_model}×{self.d_model}]",
                        font_size=self.fs, color=_hex("weight_matrix"))
        wo_title.next_to(wo_grid, UP, buff=0.12)
        self.wo_group = VGroup(wo_title, wo_grid)
        self.wo_grid = wo_grid

        # Output vector (nd rows × 1)
        def out_color(i, j):
            return _lerp(COLORS["dim"], COLORS["output"],
                         float(np.random.default_rng(i + 999).uniform(0.4, 1.0)))

        out_grid = self._color_block(nd, 1, out_color)
        out_title = Text(f"Output\n[{self.d_model}]", font_size=self.fs,
                         color=_hex("output"))
        out_title.next_to(out_grid, UP, buff=0.12)
        self.output_group = VGroup(out_title, out_grid)
        self.out_grid = out_grid

        # Symbols
        self.times_sym = MathTex(r"\times", font_size=28, color=WHITE)
        self.eq_sym    = MathTex(r"=",      font_size=28, color=WHITE)

        # Arrange
        self.input_group.move_to(ORIGIN)
        self.times_sym.next_to(self.input_group, RIGHT, buff=0.25)
        self.wo_group.next_to(self.times_sym, RIGHT, buff=0.25)
        self.eq_sym.next_to(self.wo_group, RIGHT, buff=0.25)
        self.output_group.next_to(self.eq_sym, RIGHT, buff=0.25)

        # Dimension label row
        dim_note = Text(
            f"({self.d_model}) × ({self.d_model}×{self.d_model}) = ({self.d_model})",
            font_size=self.fs - 2, color=GRAY,
        )
        dim_note.next_to(self.output_group, DOWN, buff=0.35)
        self.label_row = dim_note

        # Column highlights (for animation — initially invisible)
        self._col_highlights = []
        for j in range(nd):
            hl = SurroundingRectangle(
                VGroup(*[self.wo_grid[i][j] for i in range(nd)]),
                color=_hex("highlight"), buff=0.01, stroke_width=1.5,
            )
            hl.set_opacity(0)
            self._col_highlights.append(hl)
        self.add(*self._col_highlights)

    # ── Animation API ─────────────────────────────────────────────────────────

    def build_anim(self, run_time: float = 0.55) -> Succession:
        steps = []

        # 1. Input vector
        steps.append(FadeIn(self.input_group, shift=RIGHT * 0.1, run_time=run_time))

        # 2. × symbol + W_O matrix
        steps.append(AnimationGroup(
            FadeIn(self.times_sym),
            FadeIn(self.wo_group, shift=RIGHT * 0.1),
            lag_ratio=0.2, run_time=run_time,
        ))

        # 3. Sweep column highlights across W_O (shows matrix-vector multiply)
        n_sweep = min(self.nd, 6)  # show first few columns for clarity
        sweep_anims = []
        for j in range(n_sweep):
            hl = self._col_highlights[j]
            sweep_anims.append(
                hl.animate(run_time=run_time * 0.25).set_opacity(1)
            )
            sweep_anims.append(
                hl.animate(run_time=run_time * 0.2).set_opacity(0)
            )
        steps.append(Succession(*sweep_anims))

        # 4. = symbol + output
        steps.append(AnimationGroup(
            FadeIn(self.eq_sym),
            FadeIn(self.output_group, shift=RIGHT * 0.1),
            lag_ratio=0.2, run_time=run_time,
        ))

        # 5. Flash output
        steps.append(Flash(
            self.out_grid, color=_hex("output"),
            flash_radius=0.5, run_time=run_time * 0.6,
        ))

        # 6. Dimension note
        steps.append(FadeIn(self.label_row, run_time=run_time * 0.5))

        return Succession(*steps)

    def highlight_output(self) -> Indicate:
        return Indicate(self.output_group, color=_hex("highlight"),
                        scale_factor=1.06)

    def connect_to(self, other: VMobject) -> Arrow:
        return Arrow(
            self.output_group.get_right() + RIGHT * 0.05,
            other.get_left() + LEFT * 0.05,
            buff=0.08, color=_hex("arrow"), stroke_width=2,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Asset 5 — HeadComparisonDisplay
# ═══════════════════════════════════════════════════════════════════════════════

class HeadComparisonDisplay(VGroup):
    """
    All h head heatmaps tiled in a grid with pattern labels.

    Designed to be the *final reveal* shot that shows how different heads
    specialize on different aspects of the input.

    Parameters
    ----------
    num_heads       : number of heads
    seq_len         : sequence length
    token_labels    : token strings
    weights_list    : optional list of weight matrices (one per head)
    pattern_labels  : dict  {head_idx: "pattern name"}, shown below each head
    cell_size       : (w, h) per cell
    heads_per_row   : layout columns
    font_size       : text size
    """

    _DEFAULT_PATTERNS = [
        "Syntactic",
        "Coreference",
        "Local context",
        "Long-range dep.",
        "Subject-verb",
        "Positional",
        "Semantic sim.",
        "Rare tokens",
    ]

    def __init__(
        self,
        num_heads: int = 4,
        seq_len: int = 5,
        token_labels: Sequence[str] | None = None,
        weights_list: Sequence[Sequence[Sequence[float]]] | None = None,
        pattern_labels: dict[int, str] | None = None,
        cell_size: tuple[float, float] = (0.28, 0.28),
        heads_per_row: int = 4,
        font_size: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.token_labels = token_labels or [f"t{i}" for i in range(seq_len)]
        self.cw, self.ch = cell_size
        self.fs = font_size
        self.hpr = heads_per_row

        if weights_list is not None:
            self.weights = [np.array(w) for w in weights_list]
        else:
            self.weights = [
                _make_random_weights(seq_len, seed=h * 7 + 3)
                for h in range(num_heads)
            ]

        self.pattern_labels = pattern_labels or {
            h: self._DEFAULT_PATTERNS[h % len(self._DEFAULT_PATTERNS)]
            for h in range(num_heads)
        }

        self._build()
        self.add(self.master_title, self.layout_group, self.footer_note)

    def _single_head_card(self, h_idx: int) -> VGroup:
        """One head's card: heatmap + head title + pattern label."""
        hcolor = _head_color(h_idx)
        w_mat = self.weights[h_idx]

        # Heatmap
        cell_rows: list[list[VGroup]] = []
        grid = VGroup()
        for i in range(self.seq_len):
            rg = VGroup()
            for j in range(self.seq_len):
                p = float(w_mat[i, j])
                rect = Rectangle(
                    width=self.cw, height=self.ch,
                    fill_color=_prob_color(p, h_idx),
                    fill_opacity=0.90,
                    stroke_color=WHITE, stroke_width=0.35,
                )
                rg.add(VGroup(rect))
            rg.arrange(RIGHT, buff=0.015)
            grid.add(rg)
        grid.arrange(DOWN, buff=0.015)

        # Token labels (compact — only left and top)
        row_lbls = VGroup()
        col_lbls = VGroup()
        for k in range(self.seq_len):
            rl = Text(self.token_labels[k], font_size=self.fs - 2, color=GRAY)
            rl.next_to(grid[k], LEFT, buff=0.08)
            row_lbls.add(rl)

            cl = Text(self.token_labels[k], font_size=self.fs - 2, color=GRAY)
            cl.next_to(grid[0][k], UP, buff=0.08)
            col_lbls.add(cl)

        # Border
        border = SurroundingRectangle(
            grid, color=hcolor, buff=0.07, stroke_width=1.8,
        )

        # Titles
        head_title = Text(f"Head {h_idx}", font_size=self.fs + 2, color=hcolor)
        head_title.next_to(grid, UP, buff=0.22)

        pattern = self.pattern_labels.get(h_idx, "")
        pattern_lbl = Text(f'"{pattern}"', font_size=self.fs,
                           color=GRAY, slant="ITALIC") if pattern else VGroup()
        pattern_lbl.next_to(grid, DOWN, buff=0.14)

        return VGroup(head_title, col_lbls, grid, row_lbls, border, pattern_lbl)

    def _build(self):
        # Master title
        self.master_title = Text(
            "Multi-Head Attention — Each head learns a different pattern",
            font_size=self.fs + 5, color=WHITE,
        )

        # Cards
        self.cards: list[VGroup] = []
        for h in range(self.num_heads):
            self.cards.append(self._single_head_card(h))

        # Layout in rows
        rows = VGroup()
        for start in range(0, self.num_heads, self.hpr):
            row_grp = VGroup(*self.cards[start: start + self.hpr])
            row_grp.arrange(RIGHT, buff=0.55)
            rows.add(row_grp)
        rows.arrange(DOWN, buff=0.6)
        self.layout_group = rows

        # Footer
        self.footer_note = MathTex(
            r"\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W^O",
            font_size=self.fs + 4, color=GRAY,
        )

        # Position
        self.master_title.next_to(self.layout_group, UP, buff=0.3)
        self.footer_note.next_to(self.layout_group, DOWN, buff=0.3)

    # ── Animation API ─────────────────────────────────────────────────────────

    def build_anim(self, run_time_per_card: float = 0.45) -> Succession:
        """
        Title → cards reveal one by one (each head pops in) → formula.
        """
        steps = []

        # 1. Master title
        steps.append(Write(self.master_title, run_time=run_time_per_card))

        # 2. Cards, row by row
        for start in range(0, self.num_heads, self.hpr):
            row_cards = self.cards[start: start + self.hpr]
            steps.append(LaggedStart(
                *[FadeIn(card, scale=0.85, shift=UP * 0.1)
                  for card in row_cards],
                lag_ratio=0.2, run_time=run_time_per_card * len(row_cards),
            ))

        # 3. Footer formula
        steps.append(Write(self.footer_note, run_time=run_time_per_card))

        return Succession(*steps)

    def highlight_head(self, h_idx: int) -> AnimationGroup:
        card = self.cards[h_idx]
        border = card[4]   # SurroundingRectangle
        title  = card[0]
        return AnimationGroup(
            Indicate(border, color=_hex("highlight"), scale_factor=1.08),
            Indicate(title,  color=_hex("highlight"), scale_factor=1.1),
        )

    def zoom_to_head(self, h_idx: int, scene: Scene, run_time: float = 0.7):
        """
        Scale + shift the display so head h fills the screen.
        Useful for then showing detail from MultiHeadGrid.
        Call from scene.construct().
        """
        card = self.cards[h_idx]
        scene.play(
            self.animate
                .scale(0.45)
                .to_corner(UL, buff=0.2),
            card.animate.scale(1.8).move_to(ORIGIN),
            run_time=run_time,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Demo scene — manim -pql multi_head.py DemoScene
# ═══════════════════════════════════════════════════════════════════════════════

class DemoScene(Scene):
    """Runs all 5 assets back-to-back."""

    def construct(self):
        tokens = ["the", "cat", "sat", "on"]
        n = len(tokens)
        h = 4
        d_k = 16
        d_model = h * d_k

        # ── 1. HeadSplitAnim ──────────────────────────────────────────────────
        split = HeadSplitAnim(
            d_model=d_model, num_heads=h,
            token_label="x[cat]", font_size=13,
        )
        split.scale(0.85).move_to(ORIGIN)
        self.play(split.build_anim(run_time=0.45))
        self.play(split.highlight_head(1))
        self.wait(0.4)
        self.play(FadeOut(split))

        # ── 2. MultiHeadGrid ──────────────────────────────────────────────────
        grid = MultiHeadGrid(
            num_heads=h, seq_len=n,
            token_labels=tokens,
            cell_size=(0.38, 0.38),
            heads_per_row=4,
            font_size=11,
        )
        grid.scale(0.82).move_to(ORIGIN)
        self.play(grid.build_anim(run_time_per_head=0.40))
        self.play(grid.highlight_head(2))
        self.wait(0.4)
        self.play(FadeOut(grid))

        # ── 3. HeadConcatAnim ─────────────────────────────────────────────────
        concat = HeadConcatAnim(
            num_heads=h, d_v=d_k,
            seq_token_idx=1, font_size=13,
        )
        concat.scale(0.85).move_to(ORIGIN)
        self.play(concat.build_anim(run_time=0.50))
        self.play(concat.highlight_head_slice(0))
        self.wait(0.4)
        self.play(FadeOut(concat))

        # ── 4. OutputProjectionAnim ───────────────────────────────────────────
        proj = OutputProjectionAnim(
            d_model=d_model, num_heads=h,
            n_display=h * 4, font_size=12,
        )
        proj.scale(0.82).move_to(ORIGIN)
        self.play(proj.build_anim(run_time=0.50))
        self.play(proj.highlight_output())
        self.wait(0.4)
        self.play(FadeOut(proj))

        # ── 5. HeadComparisonDisplay ──────────────────────────────────────────
        compare = HeadComparisonDisplay(
            num_heads=h, seq_len=n,
            token_labels=tokens,
            heads_per_row=4,
            cell_size=(0.30, 0.30),
            font_size=10,
        )
        compare.scale(0.78).move_to(ORIGIN)
        self.play(compare.build_anim(run_time_per_card=0.40))
        self.play(compare.highlight_head(3))
        self.wait(0.8)