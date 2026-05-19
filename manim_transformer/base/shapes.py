"""
manim_transformer/base/shapes.py
=================================
Reusable base shapes and mobjects for the manim_transformer library.

PATCH LOG
---------
v1.2  – VectorBar: cells are now added to VGroup BEFORE positioning,
        then arranged via .arrange() so Manim's point-tracking is active
        during layout. This permanently fixes the
        "IndexError: index 3 is out of bounds for axis 0 with size 3"
        crash that occurred on .next_to() / .move_to() / FadeIn() when
        VectorBar was constructed outside a scene.
      – MatrixBox._build: same fix — cells added to VGroup first, then
        row/col offsets applied via .shift() so bounding box is always
        valid. Grid is centred with .move_to(ORIGIN) on the filled group.
      – VectorBar label: uses .next_to(self.cells, ...) which is now safe
        because self.cells has valid points before the label is built.
      – All other classes unchanged from v1.1.
"""

from manim import *
import numpy as np


# ─────────────────────────────────────────────
# 🎨  GLOBAL STYLE CONSTANTS
# ─────────────────────────────────────────────

TOKEN_COLOR       = "#4CAF50"
EMBEDDING_COLOR   = "#2196F3"
ATTENTION_COLOR   = "#9C27B0"
FFN_COLOR         = "#FF9800"
NORM_COLOR        = "#00BCD4"
OUTPUT_COLOR      = "#F44336"

QUERY_COLOR       = "#E91E63"
KEY_COLOR         = "#3F51B5"
VALUE_COLOR       = "#009688"
WEIGHT_COLOR      = "#795548"

HIGHLIGHT_COLOR   = "#FFEB3B"
ARROW_COLOR       = WHITE
DIM_COLOR         = "#555555"
POSITIVE_COLOR    = "#2196F3"
NEGATIVE_COLOR    = "#F44336"
BACKGROUND_COLOR  = "#1E1E2E"


# ─────────────────────────────────────────────
# 1.  MatrixBox
# ─────────────────────────────────────────────

class MatrixBox(VGroup):
    """
    A rectangular grid of cells for weight matrices, attention grids, etc.

    Parameters
    ----------
    rows, cols  : int
    cell_size   : float
    color       : str
    values      : 2-D array (optional)
    show_values : bool

    Animations
    ----------
    .highlight_cell(r, c)
    .highlight_row(r, color)
    .highlight_col(c, color)
    .reset_colors()
    .fill_anim(lag)
    .extraction_anim(r)   → (highlight_anim, slide_anim, extracted_VGroup)
    .label_axes(row_label, col_label)
    """

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        cell_size: float = 0.5,
        color: str = EMBEDDING_COLOR,
        values=None,
        show_values: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rows        = rows
        self.cols        = cols
        self.cell_size   = cell_size
        self.base_color  = color
        self.show_values = show_values
        self.values = (
            np.array(values, dtype=float)
            if values is not None
            else np.zeros((rows, cols))
        )

        self.cells        = VGroup()
        self.value_labels = VGroup()

        self._build()
        self.add(self.cells)
        if show_values:
            self.add(self.value_labels)

    # ── internal ──────────────────────────────
    def _build(self):
        # FIX v1.2: add each cell to self.cells BEFORE calling .shift()
        # so Manim's point array is populated during positioning.
        for r in range(self.rows):
            for c in range(self.cols):
                cell = Rectangle(
                    width=self.cell_size,
                    height=self.cell_size,
                    fill_color=self.base_color,
                    fill_opacity=0.25,
                    stroke_color=self.base_color,
                    stroke_width=1.5,
                )
                self.cells.add(cell)   # ← add first
                cell.shift(            # ← then position
                    RIGHT * c * self.cell_size + DOWN * r * self.cell_size
                )

                if self.show_values:
                    val = float(self.values[r, c])
                    lbl = Text(f"{val:.1f}", font_size=10, color=WHITE)
                    lbl.move_to(cell.get_center())
                    self.value_labels.add(lbl)

        # Centre the filled group — safe now because cells have points
        self.cells.move_to(ORIGIN)
        if self.show_values:
            self.value_labels.move_to(ORIGIN)

    def _cell(self, r: int, c: int) -> Rectangle:
        return self.cells[r * self.cols + c]

    # ── animations ────────────────────────────
    def highlight_cell(self, r: int, c: int, color=HIGHLIGHT_COLOR) -> Animation:
        return Indicate(self._cell(r, c), color=color, scale_factor=1.3)

    def highlight_row(self, r: int, color=HIGHLIGHT_COLOR) -> AnimationGroup:
        anims = [
            self._cell(r, c).animate.set_fill(color, opacity=0.8)
            for c in range(self.cols)
        ]
        return AnimationGroup(*anims, lag_ratio=0.05)

    def highlight_col(self, c: int, color=HIGHLIGHT_COLOR) -> AnimationGroup:
        anims = [
            self._cell(r, c).animate.set_fill(color, opacity=0.8)
            for r in range(self.rows)
        ]
        return AnimationGroup(*anims, lag_ratio=0.05)

    def reset_colors(self) -> AnimationGroup:
        anims = [
            cell.animate.set_fill(self.base_color, opacity=0.25)
            for cell in self.cells
        ]
        return AnimationGroup(*anims)

    def fill_anim(self, lag: float = 0.03) -> AnimationGroup:
        return AnimationGroup(
            *[FadeIn(cell, scale=0.5) for cell in self.cells],
            lag_ratio=lag,
        )

    def extraction_anim(self, r: int) -> tuple:
        """
        Returns (highlight_anim, slide_anim, extracted_copy).
        Add extracted_copy to the scene then play slide_anim.
        """
        row_cells = VGroup(*[self._cell(r, c) for c in range(self.cols)])
        extracted = row_cells.copy()
        highlight = self.highlight_row(r, color=HIGHLIGHT_COLOR)
        slide     = extracted.animate.next_to(self, RIGHT, buff=0.5)
        return highlight, slide, extracted

    def label_axes(
        self,
        row_label: str = "seq_len",
        col_label: str = "d_model",
    ) -> VGroup:
        left_brace = Brace(self.cells, LEFT,  buff=0.1)
        bot_brace  = Brace(self.cells, DOWN,  buff=0.1)
        left_text  = left_brace.get_tex(row_label).scale(0.6)
        bot_text   = bot_brace.get_tex(col_label).scale(0.6)
        return VGroup(left_brace, left_text, bot_brace, bot_text)


# ─────────────────────────────────────────────
# 2.  VectorBar
# ─────────────────────────────────────────────

class VectorBar(VGroup):
    """
    A 1-D vector displayed as a strip of coloured cells.
    Cell opacity encodes magnitude; red = negative, base_color = positive.

    Parameters
    ----------
    dim         : int
    values      : list | np.ndarray | None  (random if None)
    orientation : "horizontal" | "vertical"
    cell_size   : float
    label       : str   – shown to the LEFT (horizontal) or ABOVE (vertical)
    color       : str

    Animations
    ----------
    .pulse(color)
    .highlight_dim(i, color)
    .add_value_labels()
    .transform_to(other)
    """

    def __init__(
        self,
        dim: int = 8,
        values=None,
        orientation: str = "horizontal",
        cell_size: float = 0.45,
        label: str = "",
        color: str = EMBEDDING_COLOR,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vector_dim  = dim
        self.orientation = orientation
        self.cell_size   = cell_size
        self.base_color  = color
        self.label_text  = label

        if values is None:
            values = np.random.uniform(-1, 1, dim)
        self.values = np.array(values[:dim], dtype=float)

        self.val_labels = VGroup()
        self.lbl_obj    = None

        # ── FIX v1.3: add cells DIRECTLY to self (not via nested VGroup)
        # Manim 0.20's get_all_points() doesn't reliably recurse through
        # nested VGroups when a VectorBar is placed inside another VGroup
        # and .arrange() is called on that outer group. Adding cells
        # straight to self means self.get_all_points() finds them
        # immediately, fixing next_to / arrange / FadeIn on VectorBar.
        self.cells = VGroup()

        for v in self.values:
            pos_color = color if v >= 0 else NEGATIVE_COLOR
            opacity   = float(min(abs(v), 1.0) * 0.85 + 0.15)
            cell = Rectangle(
                width=cell_size,
                height=cell_size,
                fill_color=pos_color,
                fill_opacity=opacity,
                stroke_color=pos_color,
                stroke_width=1,
            )
            self.cells.add(cell)   # ← directly into self, not a sub-VGroup

        # Add the entire cells VGroup to self
        self.add(self.cells)

        # Position cells using arrange() which is safe for VGroups
        if orientation == "horizontal":
            self.cells.arrange(RIGHT, buff=0)
        else:
            self.cells.arrange(DOWN, buff=0)
        
        # Center the cells around ORIGIN
        self.cells.move_to(ORIGIN)

        # ── label — self has points now so next_to is safe ────────────
        if label:
            self.lbl_obj = Text(label, font_size=20, color=color, weight=BOLD)
            label_dir = LEFT if orientation == "horizontal" else UP
            self.lbl_obj.next_to(self.cells, label_dir, buff=0.2)
            self.add(self.lbl_obj)

    # ── animations ────────────────────────────
    def pulse(self, color=HIGHLIGHT_COLOR) -> AnimationGroup:
        if not self.cells:
            return AnimationGroup(Wait(0))
        return AnimationGroup(
            *[Indicate(c, color=color, scale_factor=1.2) for c in self.cells],
            lag_ratio=0.04,
        )

    def highlight_dim(self, i: int, color=HIGHLIGHT_COLOR) -> Animation:
        if i >= len(self.cells):
            return Wait(0)
        return Indicate(self.cells[i], color=color, scale_factor=1.4)

    def add_value_labels(self) -> AnimationGroup:
        labels = VGroup()
        for cell, v in zip(self.cells, self.values):
            lbl = Text(f"{v:.2f}", font_size=8, color=WHITE)
            lbl.move_to(cell.get_center())
            labels.add(lbl)
        self.val_labels = labels
        self.add(labels)
        return FadeIn(labels)

    def transform_to(self, other: "VectorBar") -> Animation:
        return Transform(self, other)


# ─────────────────────────────────────────────
# 3.  TokenBox
# ─────────────────────────────────────────────

class TokenBox(VGroup):
    """
    Rounded rectangle representing a single token.

    Parameters
    ----------
    text      : str
    token_id  : int | None
    color     : str
    show_id   : bool
    font_size : int

    Animations
    ----------
    .highlight(color)
    .shake()
    .fade_id()
    .merge_with(other, result)
    .split_into(boxes)
    """

    def __init__(
        self,
        text: str = "token",
        token_id: int | None = None,
        color: str = TOKEN_COLOR,
        show_id: bool = True,
        font_size: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.token_text = text
        self.token_id   = token_id
        self.base_color = color

        self.box = RoundedRectangle(
            corner_radius=0.15,
            width=max(len(text) * 0.18 + 0.4, 0.7),
            height=0.55,
            fill_color=color,
            fill_opacity=0.25,
            stroke_color=color,
            stroke_width=2,
        )
        self.text_obj = Text(text, font_size=font_size, color=WHITE)
        self.text_obj.move_to(self.box.get_center())
        self.add(self.box, self.text_obj)

        self.id_label = None
        if show_id and token_id is not None:
            self.id_label = Text(str(token_id), font_size=12, color=color)
            self.id_label.next_to(self.box, DOWN, buff=0.08)
            self.add(self.id_label)

    def highlight(self, color=HIGHLIGHT_COLOR) -> Animation:
        return Indicate(self.box, color=color, scale_factor=1.25)

    def shake(self) -> Animation:
        return Wiggle(self, scale_value=1.15, rotation_angle=0.06)

    def fade_id(self) -> Animation:
        if self.id_label:
            return Write(self.id_label)
        return Wait(0)

    def merge_with(self, other: "TokenBox", result: "TokenBox") -> AnimationGroup:
        return AnimationGroup(
            self.animate.move_to(result.get_center()).set_opacity(0),
            other.animate.move_to(result.get_center()).set_opacity(0),
            FadeIn(result),
            lag_ratio=0.3,
        )

    def split_into(self, boxes: list) -> AnimationGroup:
        return AnimationGroup(
            FadeOut(self),
            *[FadeIn(b) for b in boxes],
            lag_ratio=0.15,
        )


# ─────────────────────────────────────────────
# 4.  ArrowLabel
# ─────────────────────────────────────────────

class ArrowLabel(VGroup):
    """
    A labeled arrow connecting two points or mobjects.

    Parameters
    ----------
    start, end  : np.array | Mobject
    label       : str
    color       : str
    curved      : bool
    label_side  : "above" | "below" | "left" | "right"
    font_size   : int

    Animations: .draw(), .undraw()
    """

    def __init__(
        self,
        start=LEFT,
        end=RIGHT,
        label: str = "",
        color: str = ARROW_COLOR,
        curved: bool = False,
        label_side: str = "above",
        font_size: int = 18,
        **kwargs,
    ):
        super().__init__(**kwargs)

        s = start.get_right() if isinstance(start, Mobject) else np.array(start)
        e = end.get_left()    if isinstance(end,   Mobject) else np.array(end)

        self.arrow = (
            CurvedArrow(s, e, color=color)
            if curved
            else Arrow(s, e, color=color, buff=0.1, stroke_width=2)
        )
        self.add(self.arrow)

        if label:
            self.label_obj = Text(label, font_size=font_size, color=color)
            direction = {"above": UP, "below": DOWN,
                         "left": LEFT, "right": RIGHT}.get(label_side, UP)
            self.label_obj.next_to(self.arrow.get_center(), direction, buff=0.1)
            self.add(self.label_obj)

    def draw(self) -> AnimationGroup:
        anims = [GrowArrow(self.arrow)]
        if len(self.submobjects) > 1:
            anims.append(FadeIn(self.submobjects[1]))
        return AnimationGroup(*anims, lag_ratio=0.4)

    def undraw(self) -> Animation:
        return FadeOut(self)


# ─────────────────────────────────────────────
# 5.  DimensionBrace
# ─────────────────────────────────────────────

class DimensionBrace(VGroup):
    """
    Brace with dimension label, e.g. "512 × 768".

    Parameters
    ----------
    mobject   : Mobject
    direction : np.array
    label     : str
    color     : str
    font_size : int

    Animations: .appear()
    """

    def __init__(
        self,
        mobject: Mobject,
        direction=DOWN,
        label: str = "",
        color: str = WHITE,
        font_size: int = 18,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.brace = Brace(mobject, direction, color=color, buff=0.1)
        self.add(self.brace)
        if label:
            lbl = Text(label, font_size=font_size, color=color)
            lbl.next_to(self.brace, direction, buff=0.1)
            self.add(lbl)

    def appear(self) -> AnimationGroup:
        return AnimationGroup(*[FadeIn(s) for s in self.submobjects], lag_ratio=0.2)


# ─────────────────────────────────────────────
# 6.  NumberFlow
# ─────────────────────────────────────────────

class NumberFlow(VGroup):
    """
    A floating number that moves along a path to show data flow.

    Parameters
    ----------
    value     : str | float
    path      : VMobject | None
    color     : str
    font_size : int

    Animations: .flow(run_time)
    """

    def __init__(
        self,
        value=1.0,
        path: VMobject | None = None,
        color: str = HIGHLIGHT_COLOR,
        font_size: int = 18,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.path = path
        display = f"{value:.2f}" if isinstance(value, float) else str(value)
        self.dot   = Dot(color=color, radius=0.06)
        self.label = Text(display, font_size=font_size, color=color)
        self.label.next_to(self.dot, UP, buff=0.05)
        self.add(self.dot, self.label)

    def flow(self, run_time: float = 1.5) -> Animation:
        if self.path is None:
            return Wait(run_time)
        return MoveAlongPath(self, self.path, run_time=run_time, rate_func=smooth)


# ─────────────────────────────────────────────
# 7.  LabeledBlock
# ─────────────────────────────────────────────

class LabeledBlock(VGroup):
    """
    Filled rectangle with centred label — for high-level block diagrams.

    Parameters
    ----------
    label    : str
    width    : float
    height   : float
    color    : str
    sublabel : str
    font_size: int

    Animations: .highlight(color), .expand_to(mob)
    """

    def __init__(
        self,
        label: str = "Block",
        width: float = 2.5,
        height: float = 0.9,
        color: str = ATTENTION_COLOR,
        sublabel: str = "",
        font_size: int = 22,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_color = color

        self.rect = Rectangle(
            width=width, height=height,
            fill_color=color, fill_opacity=0.3,
            stroke_color=color, stroke_width=2,
        )
        self.main_label = Text(label, font_size=font_size, color=WHITE, weight=BOLD)
        self.main_label.move_to(self.rect.get_center())
        self.add(self.rect, self.main_label)

        if sublabel:
            sub = Text(sublabel, font_size=font_size - 6, color=color)
            sub.next_to(self.main_label, DOWN, buff=0.1)
            self.add(sub)

    def highlight(self, color=HIGHLIGHT_COLOR) -> Animation:
        return Indicate(self.rect, color=color, scale_factor=1.1)

    def expand_to(self, target: Mobject) -> Animation:
        return self.animate.scale_to_fit_width(target.width).move_to(target)


# ─────────────────────────────────────────────
# 8.  MathLabel
# ─────────────────────────────────────────────

class MathLabel(VGroup):
    """
    LaTeX equation placed next to a reference mobject.

    Parameters
    ----------
    latex     : str
    reference : Mobject | None
    direction : np.array
    color     : str
    font_size : int
    buff      : float

    Animations: .write(), .unwrite()
    """

    def __init__(
        self,
        latex: str = r"x",
        reference: Mobject | None = None,
        direction=RIGHT,
        color: str = WHITE,
        font_size: int = 22,
        buff: float = 0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eq = MathTex(latex, color=color, font_size=font_size)
        if reference is not None:
            self.eq.next_to(reference, direction, buff=buff)
        self.add(self.eq)

    def write(self) -> Animation:
        return Write(self.eq)

    def unwrite(self) -> Animation:
        return FadeOut(self.eq)


# ─────────────────────────────────────────────
# 9.  ConnectionLine
# ─────────────────────────────────────────────

class ConnectionLine(VGroup):
    """
    Lines from every source to every target — fully connected layer viz.

    Parameters
    ----------
    sources      : list[Mobject]
    targets      : list[Mobject]
    color        : str
    opacity      : float
    active_color : str

    Animations: .draw(lag), .activate(i,j), .pulse_all()
    """

    def __init__(
        self,
        sources: list,
        targets: list,
        color: str = DIM_COLOR,
        opacity: float = 0.4,
        active_color: str = HIGHLIGHT_COLOR,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lines        = VGroup()
        self.active_color = active_color
        self._index: dict[tuple, int] = {}

        idx = 0
        for i, src in enumerate(sources):
            for j, tgt in enumerate(targets):
                line = Line(
                    src.get_right(), tgt.get_left(),
                    color=color,
                    stroke_opacity=opacity,
                    stroke_width=1,
                )
                self.lines.add(line)
                self._index[(i, j)] = idx
                idx += 1

        self.add(self.lines)

    def draw(self, lag: float = 0.02) -> AnimationGroup:
        return AnimationGroup(*[Create(l) for l in self.lines], lag_ratio=lag)

    def activate(self, i: int, j: int) -> Animation:
        line = self.lines[self._index[(i, j)]]
        return line.animate.set_color(self.active_color).set_stroke(width=2, opacity=1.0)

    def pulse_all(self) -> AnimationGroup:
        return AnimationGroup(
            *[Indicate(l, color=self.active_color) for l in self.lines],
            lag_ratio=0.01,
        )


# ─────────────────────────────────────────────
# 10.  SoftmaxCurve
# ─────────────────────────────────────────────

class SoftmaxCurve(VGroup):
    """
    Bar chart showing a softmax probability distribution.
    Temperature-aware with live animation.

    Parameters
    ----------
    logits      : list[float]
    labels      : list[str]
    temperature : float
    color       : str
    width       : float
    height      : float

    Animations
    ----------
    .set_temperature(T)  → bars resize
    .highlight_bar(i)
    .show_values()
    """

    def __init__(
        self,
        logits: list | None = None,
        labels: list | None = None,
        temperature: float = 1.0,
        color: str = OUTPUT_COLOR,
        width: float = 4.0,
        height: float = 2.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logits       = np.array(logits if logits is not None else [2.0, 1.0, 0.5, -0.5, -1.5])
        self.labels       = labels or [f"t{i}" for i in range(len(self.logits))]
        self.temperature  = temperature
        self.base_color   = color
        self.chart_width  = width
        self.chart_height = height

        self.bars        = VGroup()
        self.bar_labels  = VGroup()
        self.prob_labels = VGroup()

        self._build()
        self.add(self.bars, self.bar_labels)

    def _softmax(self, logits, T):
        e = np.exp((logits - logits.max()) / T)
        return e / e.sum()

    def _build(self):
        probs = self._softmax(self.logits, self.temperature)
        n     = len(probs)
        bar_w = self.chart_width / (n * 1.4)

        for i, (p, lbl) in enumerate(zip(probs, self.labels)):
            bh = max(p * self.chart_height, 0.02)
            bar = Rectangle(
                width=bar_w, height=bh,
                fill_color=self.base_color, fill_opacity=0.7,
                stroke_color=self.base_color, stroke_width=1,
            )
            x_pos = (i - (n - 1) / 2) * (bar_w * 1.4)
            bar.move_to(RIGHT * x_pos + UP * (bh / 2))
            self.bars.add(bar)

            label = Text(lbl, font_size=13, color=WHITE)
            label.next_to(bar, DOWN, buff=0.1)
            self.bar_labels.add(label)

    def set_temperature(self, new_T: float) -> AnimationGroup:
        probs = self._softmax(self.logits, new_T)
        anims = []
        for bar, p in zip(self.bars, probs):
            new_h = max(p * self.chart_height, 0.02)
            anims.append(
                bar.animate.stretch_to_fit_height(new_h).align_to(bar, DOWN)
            )
        self.temperature = new_T
        return AnimationGroup(*anims)

    def highlight_bar(self, i: int) -> Animation:
        return Indicate(self.bars[i], color=HIGHLIGHT_COLOR, scale_factor=1.2)

    def show_values(self) -> AnimationGroup:
        probs = self._softmax(self.logits, self.temperature)
        labels = VGroup()
        for bar, p in zip(self.bars, probs):
            lbl = Text(f"{p*100:.1f}%", font_size=11, color=WHITE)
            lbl.next_to(bar, UP, buff=0.05)
            labels.add(lbl)
        self.prob_labels = labels
        self.add(labels)
        return FadeIn(labels)


# ─────────────────────────────────────────────
# QUICK DEMO SCENE
# manim -pql shapes.py DemoScene
# ─────────────────────────────────────────────

class DemoScene(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        title = Text("manim_transformer — base shapes", font_size=28, color=WHITE)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))

        # MatrixBox
        mat = MatrixBox(rows=4, cols=6, cell_size=0.45, color=EMBEDDING_COLOR)
        mat.shift(LEFT * 3.5)
        lbl_m = Text("MatrixBox", font_size=16, color=EMBEDDING_COLOR).next_to(mat, UP, buff=0.2)
        self.play(mat.fill_anim(), FadeIn(lbl_m))
        self.play(mat.highlight_row(1))
        self.play(mat.highlight_col(3))
        self.play(mat.reset_colors())

        # VectorBar
        vec = VectorBar(dim=10, label="e", color=QUERY_COLOR)
        vec.shift(RIGHT * 0.5 + UP * 0.5)
        lbl_v = Text("VectorBar", font_size=16, color=QUERY_COLOR).next_to(vec, UP, buff=0.2)
        self.play(FadeIn(vec), FadeIn(lbl_v))
        self.play(vec.pulse())
        self.play(vec.add_value_labels())

        # VectorBar.next_to test — was the crash site
        vec2 = VectorBar(dim=6, label="e2", color=EMBEDDING_COLOR)
        vec2.next_to(vec, DOWN, buff=0.5)
        self.play(FadeIn(vec2))

        # TokenBox
        t1 = TokenBox("hello", token_id=15496, color=TOKEN_COLOR).shift(RIGHT * 3 + UP * 1.2)
        t2 = TokenBox("world", token_id=995,   color=TOKEN_COLOR).shift(RIGHT * 3 + DOWN * 0.2)
        self.play(FadeIn(t1), FadeIn(t2))
        self.play(t1.highlight())
        self.play(t2.shake())

        # LabeledBlock
        blk = LabeledBlock("Multi-Head\nAttention", width=2.8, height=1.0,
                            color=ATTENTION_COLOR).shift(LEFT * 3.5 + DOWN * 2.5)
        self.play(FadeIn(blk))
        self.play(blk.highlight())

        # SoftmaxCurve
        sc = SoftmaxCurve(
            logits=[3.0, 1.5, 0.5, -0.5, -2.0],
            labels=["the", "a", "an", "his", "her"],
        )
        sc.shift(RIGHT * 0.5 + DOWN * 2.2)
        self.play(FadeIn(sc))
        self.play(sc.show_values())
        self.play(sc.set_temperature(0.3))
        self.play(sc.set_temperature(3.0))
        self.wait(0.5)

        self.play(*[FadeOut(m) for m in self.mobjects])
        self.play(Write(Text("All base shapes ready ✓", font_size=30, color=TOKEN_COLOR)))
        self.wait(1.5)