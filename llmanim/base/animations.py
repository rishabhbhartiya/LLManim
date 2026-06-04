"""
llmanim/base/animations.py
=====================================
Reusable animation functions for the llmanim library.
These are standalone functions (not classes) — call them inside self.play().

Every function returns an Animation or AnimationGroup so you can:
    self.play(matrix_multiply_anim(A, B, C))
    self.play(flow_through(token, arrow))
    self.play(highlight_sequence([box1, box2, box3]))

Usage:
    from llmanim.base.animations import (
        flow_through, matrix_multiply_anim, highlight_sequence,
        fade_label, pulse_glow, data_flow_arrow,
        morph_matrix, ripple_through, typewriter,
        forward_pass_pulse, attention_flow,
        dimension_change_anim, equation_reveal,
        token_stream_anim, layer_stack_anim,
    )
"""

from manim import *
import numpy as np
from .shapes import (
    MatrixBox, VectorBar, TokenBox, ArrowLabel,
    HIGHLIGHT_COLOR, ARROW_COLOR, DIM_COLOR,
    TOKEN_COLOR, EMBEDDING_COLOR, ATTENTION_COLOR,
    FFN_COLOR, QUERY_COLOR, KEY_COLOR, VALUE_COLOR,
    BACKGROUND_COLOR, OUTPUT_COLOR,
)


# ═══════════════════════════════════════════════════════════
# 1.  flow_through
#     Move a mobject along a path (arrow / curve).
# ═══════════════════════════════════════════════════════════

def flow_through(
    mob: Mobject,
    path: VMobject,
    run_time: float = 1.2,
    fade_out: bool = True,
    rate_func=smooth,
) -> AnimationGroup:
    """
    Slide `mob` along `path`, then optionally fade it out.

    Parameters
    ----------
    mob       : Mobject    – the object to move (e.g. NumberFlow, Dot, TokenBox copy)
    path      : VMobject   – any VMobject whose points define the path
    run_time  : float      – seconds for the travel
    fade_out  : bool       – fade mob at the end of the path
    rate_func : callable   – easing function

    Example
    -------
    arrow = Arrow(LEFT * 3, RIGHT * 3)
    dot   = Dot(color=YELLOW)
    self.play(flow_through(dot, arrow))
    """
    travel = MoveAlongPath(mob, path, run_time=run_time, rate_func=rate_func)
    if fade_out:
        return AnimationGroup(
            travel,
            UpdateFromAlphaFunc(
                mob,
                lambda m, a: m.set_opacity(1 - a if a > 0.75 else 1),
            ),
        )
    return AnimationGroup(travel)


# ═══════════════════════════════════════════════════════════
# 2.  matrix_multiply_anim
#     Step-by-step matrix multiplication visualisation.
#     Highlights row i of A, col j of B, fills cell (i,j) of C.
# ═══════════════════════════════════════════════════════════

def matrix_multiply_anim(
    A: MatrixBox,
    B: MatrixBox,
    C: MatrixBox,
    scene: Scene,
    row_color: str = QUERY_COLOR,
    col_color: str = KEY_COLOR,
    result_color: str = HIGHLIGHT_COLOR,
    cell_run_time: float = 0.18,
    show_all_cells: bool = True,
) -> None:
    """
    Animate A × B = C cell by cell.
    NOTE: this function drives the scene directly (calls scene.play)
    because it requires sequential per-cell animations.

    Parameters
    ----------
    A, B, C         : MatrixBox  – must have the same compatible dimensions
    scene           : Scene      – the Manim scene to play into
    row_color       : str        – highlight color for A rows
    col_color       : str        – highlight color for B cols
    result_color    : str        – highlight color for C cells being filled
    cell_run_time   : float      – seconds per cell
    show_all_cells  : bool       – if False only animates first row (faster preview)

    Example
    -------
    matrix_multiply_anim(mat_A, mat_B, mat_C, self)
    """
    rows = A.rows
    cols = B.cols

    # operator label ×
    times = Text("×", font_size=28, color=WHITE)
    times.move_to(midpoint(A.get_right(), B.get_left()))
    equals = Text("=", font_size=28, color=WHITE)
    equals.move_to(midpoint(B.get_right(), C.get_left()))
    scene.play(FadeIn(times), FadeIn(equals))

    row_range = range(rows) if show_all_cells else range(1)
    for i in row_range:
        for j in range(cols):
            anims = [
                A.highlight_row(i, color=row_color),
                B.highlight_col(j, color=col_color),
                C.highlight_cell(i, j, color=result_color),
            ]
            scene.play(AnimationGroup(*anims, lag_ratio=0.15), run_time=cell_run_time)

    scene.play(
        A.reset_colors(),
        B.reset_colors(),
        FadeOut(times),
        FadeOut(equals),
    )


# ═══════════════════════════════════════════════════════════
# 3.  highlight_sequence
#     Light up a list of mobjects one by one.
# ═══════════════════════════════════════════════════════════

def highlight_sequence(
    mobjects: list,
    color: str = HIGHLIGHT_COLOR,
    scale: float = 1.2,
    lag: float = 0.12,
    run_time: float = 0.25,
) -> AnimationGroup:
    """
    Flash each mobject in `mobjects` sequentially.
    Great for showing token-by-token processing.

    Parameters
    ----------
    mobjects : list[Mobject]
    color    : str    – flash color
    scale    : float  – how much each item scales on flash
    lag      : float  – delay between flashes (seconds)
    run_time : float  – duration per individual flash

    Example
    -------
    token_boxes = [TokenBox("the"), TokenBox("cat"), TokenBox("sat")]
    self.play(highlight_sequence(token_boxes))
    """
    anims = [
        Indicate(m, color=color, scale_factor=scale, run_time=run_time)
        for m in mobjects
    ]
    return AnimationGroup(*anims, lag_ratio=lag)


# ═══════════════════════════════════════════════════════════
# 4.  fade_label
#     Fade in a text / equation, hold, then fade out.
# ═══════════════════════════════════════════════════════════

def fade_label(
    text: str,
    position=ORIGIN,
    font_size: int = 24,
    color: str = WHITE,
    is_latex: bool = False,
    hold_time: float = 1.5,
    scene: Scene = None,
) -> Mobject:
    """
    Create a text/equation label, fade it in, and schedule fade-out.
    Returns the mobject so you can keep a reference.

    Parameters
    ----------
    text      : str     – display string or LaTeX
    position  : array   – where to place it
    font_size : int
    color     : str
    is_latex  : bool    – use MathTex instead of Text
    hold_time : float   – how long it stays visible (scene.wait)
    scene     : Scene   – if provided, plays fade-in/out automatically

    Example
    -------
    fade_label(r"\\text{softmax}(QK^T/\\sqrt{d_k})", is_latex=True, scene=self)
    """
    if is_latex:
        label = MathTex(text, font_size=font_size, color=color)
    else:
        label = Text(text, font_size=font_size, color=color)
    label.move_to(position)

    if scene is not None:
        scene.play(FadeIn(label))
        scene.wait(hold_time)
        scene.play(FadeOut(label))

    return label


# ═══════════════════════════════════════════════════════════
# 5.  pulse_glow
#     Radiate a glow ring outward from a mobject.
# ═══════════════════════════════════════════════════════════

def pulse_glow(
    mob: Mobject,
    color: str = HIGHLIGHT_COLOR,
    n_pulses: int = 2,
    max_scale: float = 1.6,
    run_time: float = 0.7,
) -> AnimationGroup:
    """
    Send n_pulses expanding glow rings out from `mob`.
    Used to emphasise important steps (key insight moments).

    Parameters
    ----------
    mob       : Mobject – the object to pulse around
    color     : str     – glow color
    n_pulses  : int     – how many rings
    max_scale : float   – how far the ring expands
    run_time  : float   – seconds per pulse

    Example
    -------
    self.play(pulse_glow(attention_matrix, color=ATTENTION_COLOR))
    """
    pulses = []
    for i in range(n_pulses):
        ring = mob.copy()
        ring.set_fill(opacity=0)
        ring.set_stroke(color=color, width=3, opacity=0.9)
        pulses.append(
            Succession(
                Wait(i * run_time * 0.4),
                AnimationGroup(
                    ring.animate.scale(max_scale).set_stroke(opacity=0),
                    run_time=run_time,
                    rate_func=rush_into,
                ),
            )
        )
    return AnimationGroup(*pulses)


# ═══════════════════════════════════════════════════════════
# 6.  data_flow_arrow
#     Draw an animated arrow from A to B with a data label.
# ═══════════════════════════════════════════════════════════

def data_flow_arrow(
    start: Mobject | np.ndarray,
    end:   Mobject | np.ndarray,
    label: str = "",
    color: str = ARROW_COLOR,
    label_color: str = HIGHLIGHT_COLOR,
    curved: bool = False,
    run_time: float = 0.6,
) -> tuple:
    """
    Create and return (arrow_mobject, grow_animation).
    Optionally adds a label showing what data flows through.

    Parameters
    ----------
    start, end   : Mobject or np.ndarray
    label        : str   – e.g. "(seq, d_model)"
    color        : str   – arrow color
    label_color  : str   – label color
    curved       : bool  – use CurvedArrow
    run_time     : float

    Returns
    -------
    (VGroup, AnimationGroup)  – (arrow+label mobject, play-able animation)

    Example
    -------
    arrow_mob, arrow_anim = data_flow_arrow(embed_block, attn_block, "(8, 512)")
    self.add(arrow_mob)
    self.play(arrow_anim)
    """
    s = start.get_right() if isinstance(start, Mobject) else start
    e = end.get_left()    if isinstance(end,   Mobject) else end

    if curved:
        arrow = CurvedArrow(s, e, color=color, stroke_width=2)
    else:
        arrow = Arrow(s, e, color=color, buff=0.1, stroke_width=2)

    group = VGroup(arrow)

    if label:
        lbl = Text(label, font_size=14, color=label_color)
        lbl.next_to(arrow.get_center(), UP, buff=0.12)
        group.add(lbl)
        anim = AnimationGroup(
            GrowArrow(arrow),
            FadeIn(lbl, shift=UP * 0.1),
            lag_ratio=0.4,
            run_time=run_time,
        )
    else:
        anim = GrowArrow(arrow, run_time=run_time)

    return group, anim


# ═══════════════════════════════════════════════════════════
# 7.  morph_matrix
#     Smoothly morph one MatrixBox into another
#     (dimension change, color change, value change).
# ═══════════════════════════════════════════════════════════

def morph_matrix(
    source: MatrixBox,
    target: MatrixBox,
    run_time: float = 1.0,
) -> Animation:
    """
    Transform `source` MatrixBox into `target` MatrixBox.
    Used to show dimension expansions (e.g. d_model → 4×d_model).

    Parameters
    ----------
    source, target : MatrixBox
    run_time       : float

    Example
    -------
    ffn_wide = MatrixBox(rows=4, cols=16, color=FFN_COLOR)
    self.play(morph_matrix(ffn_narrow, ffn_wide))
    """
    return Transform(source, target, run_time=run_time, rate_func=smooth)


# ═══════════════════════════════════════════════════════════
# 8.  ripple_through
#     A color wave sweeping left-to-right across a VGroup.
#     Perfect for showing a forward pass through a sequence.
# ═══════════════════════════════════════════════════════════

def ripple_through(
    mobjects: list,
    wave_color: str = HIGHLIGHT_COLOR,
    base_color: str = DIM_COLOR,
    lag: float = 0.08,
    run_time: float = 0.2,
) -> AnimationGroup:
    """
    Color wave passes through each mobject left→right.

    Parameters
    ----------
    mobjects   : list[Mobject]
    wave_color : str   – the traveling highlight color
    base_color : str   – color objects return to after wave
    lag        : float – delay between consecutive items
    run_time   : float – duration per item

    Example
    -------
    self.play(ripple_through(token_boxes))
    """
    def make_flash(mob):
        return Succession(
            mob.animate(run_time=run_time / 2).set_color(wave_color),
            mob.animate(run_time=run_time / 2).set_color(base_color),
        )

    return AnimationGroup(*[make_flash(m) for m in mobjects], lag_ratio=lag)


# ═══════════════════════════════════════════════════════════
# 9.  typewriter
#     Reveal text character by character.
#     Used for showing model output being generated.
# ═══════════════════════════════════════════════════════════

def typewriter(
    text: str,
    position=ORIGIN,
    color: str = WHITE,
    font_size: int = 28,
    char_delay: float = 0.07,
    scene: Scene = None,
) -> Text:
    """
    Typewriter effect — characters appear one at a time.
    Mimics LLM token-by-token generation.

    Parameters
    ----------
    text       : str    – the full string to reveal
    position   : array  – where to place it
    color      : str
    font_size  : int
    char_delay : float  – seconds between characters
    scene      : Scene  – if given, drives the animation automatically

    Returns
    -------
    Text mobject (final state)

    Example
    -------
    typewriter("The transformer is a sequence model.", scene=self)
    """
    full = Text(text, font_size=font_size, color=color)
    full.move_to(position)

    if scene is None:
        return full

    # reveal one character at a time using partial Text objects
    for i in range(1, len(text) + 1):
        partial = Text(text[:i], font_size=font_size, color=color)
        partial.move_to(position)
        if i == 1:
            scene.add(partial)
        else:
            scene.remove(prev)   # noqa: F821 – prev is always defined after i>=2
            scene.add(partial)
        scene.wait(char_delay)
        prev = partial           # noqa: F841

    return full


# ═══════════════════════════════════════════════════════════
# 10. forward_pass_pulse
#     Send a pulse of light through a stack of LabeledBlocks,
#     representing a forward pass through the network.
# ═══════════════════════════════════════════════════════════

def forward_pass_pulse(
    blocks: list,
    pulse_color: str = ATTENTION_COLOR,
    lag: float = 0.15,
    run_time: float = 0.35,
) -> AnimationGroup:
    """
    Flash each LabeledBlock (or any Mobject) in sequence,
    simulating data flowing upward through transformer layers.

    Parameters
    ----------
    blocks    : list[Mobject]   – ordered list (bottom to top)
    pulse_color : str
    lag       : float
    run_time  : float

    Example
    -------
    layers = [LabeledBlock(f"Layer {i}") for i in range(6)]
    self.play(forward_pass_pulse(layers))
    """
    def flash_block(b):
        return Succession(
            b.animate(run_time=run_time / 2).set_fill(pulse_color, opacity=0.7),
            b.animate(run_time=run_time / 2).set_fill(b.base_color, opacity=0.3),
        )

    return AnimationGroup(*[flash_block(b) for b in blocks], lag_ratio=lag)


# ═══════════════════════════════════════════════════════════
# 11. attention_flow
#     Draw curved arrows from token i to tokens it attends to,
#     weighted by attention scores.
# ═══════════════════════════════════════════════════════════

def attention_flow(
    token_boxes: list,
    attention_weights: np.ndarray,
    query_idx: int = 0,
    color: str = ATTENTION_COLOR,
    min_opacity: float = 0.1,
    max_stroke: float = 4.0,
    scene: Scene = None,
) -> VGroup:
    """
    Draw attention arrows from `query_idx` token to all others,
    with stroke width and opacity proportional to attention weight.

    Parameters
    ----------
    token_boxes       : list[TokenBox]  – the token mobjects in sequence
    attention_weights : np.ndarray      – 1-D array of shape (seq_len,)
                                          weights for the query token
    query_idx         : int             – which token is the query
    color             : str
    min_opacity       : float
    max_stroke        : float
    scene             : Scene           – if given, plays arrows appearing

    Returns
    -------
    VGroup of CurvedArrow objects

    Example
    -------
    weights = np.array([0.1, 0.6, 0.2, 0.05, 0.05])
    arrows  = attention_flow(token_boxes, weights, query_idx=1, scene=self)
    """
    arrows = VGroup()
    query_box = token_boxes[query_idx]

    for j, (tbox, w) in enumerate(zip(token_boxes, attention_weights)):
        if j == query_idx:
            continue
        if w < 0.01:
            continue

        opacity    = float(np.clip(w / attention_weights.max(), min_opacity, 1.0))
        stroke_w   = float(np.clip(w / attention_weights.max() * max_stroke, 0.5, max_stroke))

        arrow = CurvedArrow(
            query_box.get_top() + UP * 0.05,
            tbox.get_top()      + UP * 0.05,
            angle=-TAU / 6 if j > query_idx else TAU / 6,
            color=color,
            stroke_width=stroke_w,
            stroke_opacity=opacity,
        )
        arrows.add(arrow)

    if scene is not None:
        scene.play(
            AnimationGroup(*[Create(a) for a in arrows], lag_ratio=0.08)
        )

    return arrows


# ═══════════════════════════════════════════════════════════
# 12. dimension_change_anim
#     Show a vector/matrix changing shape with a dimension label.
#     Classic "d_model → 4×d_model → d_model" FFN expansion.
# ═══════════════════════════════════════════════════════════

def dimension_change_anim(
    mob: Mobject,
    new_width: float,
    new_height: float,
    label_before: str,
    label_after: str,
    scene: Scene,
    color: str = FFN_COLOR,
    run_time: float = 0.8,
) -> None:
    """
    Stretch `mob` to new dimensions and swap dimension labels.
    Drives the scene directly.

    Parameters
    ----------
    mob           : Mobject
    new_width     : float
    new_height    : float
    label_before  : str   – e.g. "768"
    label_after   : str   – e.g. "3072"
    scene         : Scene
    color         : str
    run_time      : float

    Example
    -------
    dimension_change_anim(ffn_rect, 5.0, 0.5, "768", "3072", self)
    """
    lbl_b = Text(label_before, font_size=18, color=color).next_to(mob, DOWN, buff=0.2)
    scene.play(FadeIn(lbl_b))
    scene.wait(0.3)

    lbl_a = Text(label_after, font_size=18, color=color).next_to(mob, DOWN, buff=0.2)
    scene.play(
        mob.animate(run_time=run_time).stretch_to_fit_width(new_width)
               .stretch_to_fit_height(new_height),
        Transform(lbl_b, lbl_a, run_time=run_time),
    )
    scene.wait(0.2)
    scene.play(FadeOut(lbl_b))


# ═══════════════════════════════════════════════════════════
# 13. equation_reveal
#     Build a multi-line equation term by term.
#     Each term fades in with a short pause, great for
#     explaining formulas step by step.
# ═══════════════════════════════════════════════════════════

def equation_reveal(
    terms: list,
    position=ORIGIN,
    direction=RIGHT,
    color: str = WHITE,
    font_size: int = 26,
    delay: float = 0.5,
    scene: Scene = None,
) -> VGroup:
    """
    Reveal a LaTeX equation one term at a time.

    Parameters
    ----------
    terms     : list[str]  – LaTeX strings for each term
                             e.g. ["Q", "K^T", "/", r"\\sqrt{d_k}"]
    position  : array      – starting position of first term
    direction : array      – direction to lay terms out (RIGHT or DOWN)
    color     : str
    font_size : int
    delay     : float      – wait between terms (seconds)
    scene     : Scene      – if given, drives animation automatically

    Returns
    -------
    VGroup of MathTex objects

    Example
    -------
    equation_reveal(
        [r"\\text{Attn}", "=", r"\\text{softmax}", r"\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)", "V"],
        scene=self
    )
    """
    term_mobs = VGroup()
    for i, t in enumerate(terms):
        mob = MathTex(t, font_size=font_size, color=color)
        if i == 0:
            mob.move_to(position)
        else:
            mob.next_to(term_mobs[-1], direction, buff=0.15)
        term_mobs.add(mob)

    if scene is not None:
        for mob in term_mobs:
            scene.play(FadeIn(mob, shift=UP * 0.1), run_time=0.3)
            scene.wait(delay)

    return term_mobs


# ═══════════════════════════════════════════════════════════
# 14. token_stream_anim
#     A stream of TokenBoxes slides in from the left,
#     used to show the input sequence entering the model.
# ═══════════════════════════════════════════════════════════

def token_stream_anim(
    token_boxes: list,
    target_positions: list,
    slide_from=LEFT * 8,
    lag: float = 0.1,
    run_time: float = 0.4,
) -> AnimationGroup:
    """
    Animate TokenBoxes sliding in from off-screen to their positions.

    Parameters
    ----------
    token_boxes      : list[TokenBox]
    target_positions : list[np.ndarray]  – final position for each box
    slide_from       : np.ndarray        – off-screen start point
    lag              : float             – stagger between tokens
    run_time         : float

    Example
    -------
    boxes     = [TokenBox(w) for w in ["The", "cat", "sat", "."]]
    positions = [RIGHT * i * 1.2 for i in range(4)]
    self.play(token_stream_anim(boxes, positions))
    """
    # position each box at slide_from first
    for box in token_boxes:
        box.move_to(slide_from)

    anims = [
        box.animate(run_time=run_time).move_to(pos)
        for box, pos in zip(token_boxes, target_positions)
    ]
    return AnimationGroup(*anims, lag_ratio=lag)


# ═══════════════════════════════════════════════════════════
# 15. layer_stack_anim
#     Stack N identical blocks vertically, each appearing
#     with a short delay — the classic "×12" transformer stack.
# ═══════════════════════════════════════════════════════════

def layer_stack_anim(
    block_template: Mobject,
    n_layers: int = 6,
    spacing: float = 0.75,
    direction=UP,
    lag: float = 0.12,
    color: str = ATTENTION_COLOR,
) -> tuple:
    """
    Create a vertical stack of `n_layers` copies of `block_template`.

    Parameters
    ----------
    block_template : Mobject  – the block to repeat
    n_layers       : int      – how many layers
    spacing        : float    – gap between blocks
    direction      : array    – stack direction (UP / DOWN)
    lag            : float    – animation lag
    color          : str      – stroke color for each block

    Returns
    -------
    (VGroup of blocks, AnimationGroup to play)

    Example
    -------
    blk = LabeledBlock("Transformer Block", color=ATTENTION_COLOR)
    stack, anim = layer_stack_anim(blk, n_layers=6)
    self.play(anim)
    """
    stack = VGroup()
    for i in range(n_layers):
        copy = block_template.copy()
        copy.move_to(block_template.get_center() + direction * i * spacing)
        stack.add(copy)

    stack.move_to(ORIGIN)

    anim = AnimationGroup(
        *[FadeIn(b, shift=direction * 0.3) for b in stack],
        lag_ratio=lag,
    )
    return stack, anim


# ═══════════════════════════════════════════════════════════
# 16. vector_addition_anim
#     Element-wise addition of two VectorBars → result.
#     Used for residual connections and PE + token embedding.
# ═══════════════════════════════════════════════════════════

def vector_addition_anim(
    vec_a: VectorBar,
    vec_b: VectorBar,
    result: VectorBar,
    scene: Scene,
    plus_color: str = WHITE,
    run_time: float = 0.8,
) -> None:
    """
    Animate vec_a + vec_b = result element-wise.
    Drives the scene directly.

    Parameters
    ----------
    vec_a, vec_b : VectorBar – the two input vectors
    result       : VectorBar – pre-built result vector (add to scene first, hidden)
    scene        : Scene
    plus_color   : str
    run_time     : float

    Example
    -------
    tok_emb = VectorBar(dim=8, label="token", color=EMBEDDING_COLOR)
    pos_enc = VectorBar(dim=8, label="PE",    color=NORM_COLOR)
    combined= VectorBar(dim=8, label="x",     color=QUERY_COLOR)
    scene.add(tok_emb.shift(LEFT*2), pos_enc, combined.shift(RIGHT*2).set_opacity(0))
    vector_addition_anim(tok_emb, pos_enc, combined, scene)
    """
    plus  = Text("+", font_size=32, color=plus_color)
    equals = Text("=", font_size=32, color=plus_color)
    plus.move_to(midpoint(vec_a.get_right(), vec_b.get_left()))
    equals.move_to(midpoint(vec_b.get_right(), result.get_left()))

    scene.play(FadeIn(plus), FadeIn(equals))

    # cell-by-cell glow
    scene.play(
        AnimationGroup(
            *[
                AnimationGroup(
                    Indicate(ca, color=HIGHLIGHT_COLOR),
                    Indicate(cb, color=HIGHLIGHT_COLOR),
                    FadeIn(cr),
                    lag_ratio=0.2,
                )
                for ca, cb, cr in zip(vec_a.cells, vec_b.cells, result.cells)
            ],
            lag_ratio=0.07,
            run_time=run_time,
        )
    )
    scene.play(FadeOut(plus), FadeOut(equals))


# ═══════════════════════════════════════════════════════════
# 17. softmax_temperature_sweep
#     Animate the softmax distribution changing as
#     temperature sweeps from T_start to T_end.
# ═══════════════════════════════════════════════════════════

def softmax_temperature_sweep(
    softmax_curve,          # SoftmaxCurve instance from shapes.py
    t_start: float = 1.0,
    t_end:   float = 0.2,
    steps:   int   = 8,
    scene:   Scene = None,
    run_time_per_step: float = 0.3,
) -> None:
    """
    Smoothly animate softmax bars as temperature changes.
    Drives scene directly.

    Parameters
    ----------
    softmax_curve       : SoftmaxCurve
    t_start, t_end      : float  – temperature range
    steps               : int    – number of intermediate steps
    scene               : Scene
    run_time_per_step   : float

    Example
    -------
    sc = SoftmaxCurve(logits=[3,1,0,-1,-2])
    self.add(sc)
    softmax_temperature_sweep(sc, t_start=2.0, t_end=0.1, scene=self)
    """
    if scene is None:
        return

    temps = np.linspace(t_start, t_end, steps)
    t_label = Text(f"T = {t_start:.1f}", font_size=18, color=OUTPUT_COLOR)
    t_label.next_to(softmax_curve, UP, buff=0.2)
    scene.play(FadeIn(t_label))

    for T in temps[1:]:
        new_lbl = Text(f"T = {T:.2f}", font_size=18, color=OUTPUT_COLOR)
        new_lbl.next_to(softmax_curve, UP, buff=0.2)
        scene.play(
            softmax_curve.set_temperature(T),
            Transform(t_label, new_lbl),
            run_time=run_time_per_step,
        )

    scene.wait(0.5)
    scene.play(FadeOut(t_label))


# ═══════════════════════════════════════════════════════════
# 18. zoom_into
#     Scale up and center a mobject (zoom in effect),
#     then zoom back out.  Used to dive into a sublayer.
# ═══════════════════════════════════════════════════════════

def zoom_into(
    mob: Mobject,
    scene: Scene,
    scale: float = 2.5,
    run_time: float = 0.7,
    hold_time: float = 1.0,
) -> None:
    """
    Zoom into `mob`, hold, then zoom back out.
    Everything else in the scene fades while zoomed.

    Parameters
    ----------
    mob       : Mobject
    scene     : Scene
    scale     : float  – how much to zoom in
    run_time  : float
    hold_time : float

    Example
    -------
    zoom_into(attention_block, self)
    """
    others = [m for m in scene.mobjects if m is not mob]
    scene.play(
        mob.animate(run_time=run_time).scale(scale).move_to(ORIGIN),
        *[m.animate(run_time=run_time).set_opacity(0.1) for m in others],
    )
    scene.wait(hold_time)
    scene.play(
        mob.animate(run_time=run_time).scale(1 / scale),
        *[m.animate(run_time=run_time).set_opacity(1.0) for m in others],
    )


# ═══════════════════════════════════════════════════════════
# 19. label_appear
#     A small tooltip-style label pops up above a mobject.
# ═══════════════════════════════════════════════════════════

def label_appear(
    mob: Mobject,
    text: str,
    color: str = HIGHLIGHT_COLOR,
    font_size: int = 16,
    direction=UP,
    buff: float = 0.2,
) -> tuple:
    """
    Create a small label floating above (or beside) `mob`.

    Returns
    -------
    (label_mobject, FadeIn_animation)

    Example
    -------
    lbl, anim = label_appear(matrix, "W_Q  (d_model × d_k)")
    self.play(anim)
    ...
    self.play(FadeOut(lbl))
    """
    lbl = Text(text, font_size=font_size, color=color)
    lbl.next_to(mob, direction, buff=buff)
    return lbl, FadeIn(lbl, shift=direction * 0.15)


# ═══════════════════════════════════════════════════════════
# 20. component_swap
#     Cross-fade between two mobjects in the same position.
#     Used for "before/after" comparisons (e.g. Pre vs Post norm).
# ═══════════════════════════════════════════════════════════

def component_swap(
    old_mob: Mobject,
    new_mob: Mobject,
    run_time: float = 0.6,
) -> AnimationGroup:
    """
    Fade out `old_mob` while fading in `new_mob` at the same position.

    Parameters
    ----------
    old_mob, new_mob : Mobject
    run_time         : float

    Example
    -------
    self.play(component_swap(relu_curve, gelu_curve))
    """
    new_mob.move_to(old_mob.get_center())
    return AnimationGroup(
        FadeOut(old_mob, run_time=run_time),
        FadeIn(new_mob,  run_time=run_time, shift=UP * 0.05),
    )


# ═══════════════════════════════════════════════════════════
# QUICK DEMO SCENE
# Run: manim -pql animations.py AnimationsDemoScene
# ═══════════════════════════════════════════════════════════

class AnimationsDemoScene(Scene):
    """
    Previews all animation functions from this module.
    Not for production — purely for testing.
    Run: manim -pql animations.py AnimationsDemoScene
    """

    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        title = Text("llmanim — base animations", font_size=26, color=WHITE)
        self.play(Write(title))
        self.wait(0.4)
        self.play(title.animate.to_edge(UP).scale(0.8))

        # ── 1. highlight_sequence ─────────────
        section = Text("highlight_sequence", font_size=18, color=TOKEN_COLOR).to_edge(LEFT).shift(UP * 2)
        self.play(FadeIn(section))
        words = ["The", "transformer", "attends", "to", "all", "tokens"]
        boxes  = [TokenBox(w, color=TOKEN_COLOR) for w in words]
        row    = VGroup(*boxes).arrange(RIGHT, buff=0.15).shift(UP * 1.2)
        self.play(FadeIn(row))
        self.play(highlight_sequence(boxes))
        self.wait(0.3)

        # ── 2. ripple_through ─────────────────
        section2 = Text("ripple_through", font_size=18, color=EMBEDDING_COLOR).next_to(section, DOWN, buff=0.3).to_edge(LEFT)
        self.play(FadeIn(section2))
        self.play(ripple_through(boxes, wave_color=EMBEDDING_COLOR, base_color=WHITE))
        self.wait(0.3)

        # ── 3. data_flow_arrow ────────────────
        section3 = Text("data_flow_arrow", font_size=18, color=ATTENTION_COLOR).next_to(section2, DOWN, buff=0.3).to_edge(LEFT)
        self.play(FadeIn(section3))
        blk_a = RoundedRectangle(width=1.4, height=0.6, color=EMBEDDING_COLOR, fill_opacity=0.3).shift(LEFT * 3 + DOWN * 0.5)
        blk_b = RoundedRectangle(width=1.4, height=0.6, color=ATTENTION_COLOR, fill_opacity=0.3).shift(RIGHT * 0        + DOWN * 0.5)
        blk_c = RoundedRectangle(width=1.4, height=0.6, color=FFN_COLOR,       fill_opacity=0.3).shift(RIGHT * 3 + DOWN * 0.5)
        lbl_a = Text("Embed",   font_size=14, color=WHITE).move_to(blk_a)
        lbl_b = Text("Attn",    font_size=14, color=WHITE).move_to(blk_b)
        lbl_c = Text("FFN",     font_size=14, color=WHITE).move_to(blk_c)
        self.play(FadeIn(VGroup(blk_a, blk_b, blk_c, lbl_a, lbl_b, lbl_c)))

        arr1, anim1 = data_flow_arrow(blk_a, blk_b, "(8, 512)", color=EMBEDDING_COLOR)
        arr2, anim2 = data_flow_arrow(blk_b, blk_c, "(8, 512)", color=ATTENTION_COLOR)
        self.add(arr1, arr2)
        self.play(anim1)
        self.play(anim2)
        self.wait(0.3)

        # ── 4. equation_reveal ────────────────
        section4 = Text("equation_reveal", font_size=18, color=QUERY_COLOR).next_to(section3, DOWN, buff=0.3).to_edge(LEFT)
        self.play(FadeIn(section4))
        self.play(FadeOut(VGroup(blk_a, blk_b, blk_c, lbl_a, lbl_b, lbl_c, arr1, arr2, row)))
        eq = equation_reveal(
            [r"\text{Attn}", "=", r"\text{softmax}", r"\!\left(\frac{QK^T}{\sqrt{d_k}}\right)", "V"],
            position=DOWN * 1.5,
            color=WHITE,
            font_size=28,
            delay=0.4,
            scene=self,
        )
        self.wait(0.5)

        # ── 5. pulse_glow ─────────────────────
        section5 = Text("pulse_glow", font_size=18, color=FFN_COLOR).next_to(section4, DOWN, buff=0.3).to_edge(LEFT)
        self.play(FadeIn(section5))
        self.play(pulse_glow(eq, color=HIGHLIGHT_COLOR, n_pulses=2))
        self.wait(0.5)

        # ── Outro ─────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])
        done = Text("All animations ready ✓", font_size=30, color=TOKEN_COLOR)
        self.play(Write(done))
        self.wait(1.5)