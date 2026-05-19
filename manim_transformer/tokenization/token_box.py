"""
manim_transformer/tokenization/token_box.py
============================================
Every visual asset needed to animate the tokenization stage of an LLM.

PATCH LOG
---------
v1.1  – fix RawTextDisplay.reveal_chars(scene, delay) signature
      – add  RawTextDisplay.fade_to_tokens() stub
      – fix  SpecialTokenBox.show_tooltip / hide_tooltip aliases
      – add  ContextWindowBar.appear(), fill_up_to(), fill_next()
      – fix  TokenIDMapping standalone constructor (token+id pairs)
      – guard NORM_COLOR import (not always exported by base.shapes)
v1.2  – add TokenRow.show_all_ids(scene, delay) — drives scene directly,
        animates each token's ID label appearing one by one
"""

from __future__ import annotations
from manim import *
import numpy as np

# ── safe imports from base (guard against missing constants) ─────────────────
try:
    from base.shapes import (
        TOKEN_COLOR, EMBEDDING_COLOR, ATTENTION_COLOR,
        FFN_COLOR, HIGHLIGHT_COLOR, DIM_COLOR,
        BACKGROUND_COLOR, OUTPUT_COLOR,
    )
    try:
        from base.shapes import NORM_COLOR
    except ImportError:
        NORM_COLOR = "#00BCD4"
except ImportError:
    # Fallback palette so file works standalone
    TOKEN_COLOR      = "#4CAF50"
    EMBEDDING_COLOR  = "#2196F3"
    ATTENTION_COLOR  = "#9C27B0"
    FFN_COLOR        = "#FF9800"
    HIGHLIGHT_COLOR  = "#FFEB3B"
    DIM_COLOR        = "#555555"
    BACKGROUND_COLOR = "#0D0D1A"
    OUTPUT_COLOR     = "#F44336"
    NORM_COLOR       = "#00BCD4"

try:
    from base.utils import auto_arrange_row, DEMO_VOCAB, make_token_sequence, wrap_text
except ImportError:
    auto_arrange_row = None
    DEMO_VOCAB = {
        "[PAD]": 0, "the": 1, "cat": 2, "sat": 3, "on": 4,
        "mat": 5, "transformer": 6, "hello": 7, "[CLS]": 101,
        "[SEP]": 102, "[MASK]": 103,
    }
    def make_token_sequence():
        tokens = ["the", "cat", "sat"]
        return tokens, [DEMO_VOCAB.get(t, i) for i, t in enumerate(tokens)]
    def wrap_text(t, _w):
        return t

# ── extra palette ─────────────────────────────────────────────────────────────
TOKEN_PALETTE = [
    "#4CAF50", "#2196F3", "#E91E63", "#FF9800",
    "#9C27B0", "#00BCD4", "#FFEB3B", "#F44336",
]

SPECIAL_TOKEN_COLOR = "#90A4AE"
OOV_TOKEN_COLOR     = "#F44336"
SUBWORD_COLOR       = "#FF7043"
MERGE_COLOR         = "#FDD835"
VOCAB_ROW_COLOR     = "#1565C0"
OOV_COLOR           = "#EF5350"


def _token_color(index: int) -> str:
    return TOKEN_PALETTE[index % len(TOKEN_PALETTE)]


def midpoint(a, b):
    return (np.array(a) + np.array(b)) / 2


# ═══════════════════════════════════════════════════════════
# 1.  RawTextDisplay
# ═══════════════════════════════════════════════════════════

class RawTextDisplay(VGroup):
    """
    Displays a sentence as plain text.
    Supports character-by-character reveal and per-word highlights.

    Parameters
    ----------
    text      : str   – the full sentence
    font_size : int
    color     : str   – base text color

    Key animations
    --------------
    .reveal_chars(scene, delay)  → characters appear one by one (drives scene)
    .reveal_words(lag)           → AnimationGroup, words slide in
    .highlight_word(i, color)    → Indicate flash
    .highlight_range(i, j)       → flash word range
    .dim_all()                   → AnimationGroup
    .undim_all()                 → AnimationGroup
    .strike_through(i)           → returns Line mob
    .fade_to_tokens()            → FadeOut self (placeholder before TokenRow appears)
    """

    def __init__(
        self,
        text: str = "The transformer is all you need",
        font_size: int = 32,
        color: str = WHITE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sentence   = text
        self.font_size  = font_size
        self.base_color = color

        self.words: list[str]      = text.split()
        self.word_mobs: list[Text] = []
        self._build()

    # ── internal ──────────────────────────────
    def _build(self):
        for word in self.words:
            mob = Text(word, font_size=self.font_size, color=self.base_color)
            self.word_mobs.append(mob)
        row = VGroup(*self.word_mobs).arrange(RIGHT, buff=0.18)
        row.move_to(ORIGIN)
        self.add(row)

    # ── animations ────────────────────────────

    def reveal_chars(self, scene: Scene, delay: float = 0.04) -> None:
        """
        Drive the scene directly — characters appear one by one.
        """
        for mob in self.word_mobs:
            chars = list(mob)
            if not chars:
                scene.play(FadeIn(mob, shift=DOWN * 0.05), run_time=delay * 3)
                continue
            for char in chars:
                scene.play(FadeIn(char, shift=DOWN * 0.05), run_time=delay)

    def reveal_words(self, lag: float = 0.12) -> AnimationGroup:
        """Words slide in one by one — returns AnimationGroup."""
        return AnimationGroup(
            *[FadeIn(m, shift=UP * 0.15) for m in self.word_mobs],
            lag_ratio=lag,
        )

    def highlight_word(self, i: int, color: str = HIGHLIGHT_COLOR) -> Animation:
        return Indicate(self.word_mobs[i], color=color, scale_factor=1.3)

    def highlight_range(self, i: int, j: int, color: str = HIGHLIGHT_COLOR) -> AnimationGroup:
        return AnimationGroup(
            *[Indicate(self.word_mobs[k], color=color, scale_factor=1.2)
              for k in range(i, j + 1)],
            lag_ratio=0.05,
        )

    def dim_all(self) -> AnimationGroup:
        return AnimationGroup(
            *[m.animate.set_color(DIM_COLOR) for m in self.word_mobs]
        )

    def undim_all(self) -> AnimationGroup:
        return AnimationGroup(
            *[m.animate.set_color(self.base_color) for m in self.word_mobs]
        )

    def strike_through(self, i: int, color: str = OUTPUT_COLOR) -> Line:
        mob  = self.word_mobs[i]
        line = Line(mob.get_left(), mob.get_right(), color=color, stroke_width=2)
        line.move_to(mob.get_center())
        return line

    def fade_to_tokens(self) -> Animation:
        """Convenience fade-out used before a TokenRow appears."""
        return FadeOut(self, shift=UP * 0.15)


# ═══════════════════════════════════════════════════════════
# 2.  TokenBox
# ═══════════════════════════════════════════════════════════

class TokenBox(VGroup):
    """
    A styled rounded rectangle representing a single token.

    Parameters
    ----------
    text        : str
    token_id    : int | None
    color       : str
    font_size   : int
    show_id     : bool
    is_subword  : bool
    width       : float | None

    Key animations
    --------------
    .highlight(color)     → Indicate
    .shake()              → Wiggle
    .fade_id()            → animate ID label appearing
    .pop_in()             → scale entrance
    .pop_out()            → scale exit
    .activate(color)      → fill box
    .deactivate()         → unfill
    .set_id(new_id)       → update ID label
    .merge_with(other, result)  → merge animation
    .split_into(boxes)    → split animation
    """

    def __init__(
        self,
        text: str = "token",
        token_id: int | None = None,
        color: str = TOKEN_COLOR,
        font_size: int = 22,
        show_id: bool = True,
        is_subword: bool = False,
        width: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.token_text = text
        self.token_id   = token_id
        self.base_color = color
        self.is_subword = is_subword

        auto_w = max(len(text) * 0.19 + 0.45, 0.75)
        box_w  = width if width is not None else auto_w

        self.box = RoundedRectangle(
            corner_radius=0.15,
            width=box_w,
            height=0.60,
            fill_color=color,
            fill_opacity=0.20,
            stroke_color=color,
            stroke_width=1.5 if is_subword else 2.5,
        )
        self.text_obj = Text(text, font_size=font_size, color=WHITE)
        self.text_obj.move_to(self.box.get_center())
        self.add(self.box, self.text_obj)

        self.id_label = None
        if show_id and token_id is not None:
            self.id_label = Text(str(token_id), font_size=13, color=color)
            self.id_label.next_to(self.box, DOWN, buff=0.10)
            self.add(self.id_label)

        if is_subword:
            underline = DashedLine(
                self.box.get_left()  + RIGHT * 0.1,
                self.box.get_right() + LEFT  * 0.1,
                color=color,
                dash_length=0.05,
                stroke_width=1,
            ).next_to(self.box, DOWN, buff=0.02)
            self.add(underline)

    # ── animations ────────────────────────────
    def highlight(self, color: str = HIGHLIGHT_COLOR) -> Animation:
        return Indicate(self.box, color=color, scale_factor=1.25)

    def shake(self) -> Animation:
        return Wiggle(self, scale_value=1.15, rotation_angle=0.06)

    def pop_in(self) -> Animation:
        self.scale(0.01)
        return self.animate.scale(100)

    def pop_out(self) -> Animation:
        return self.animate.scale(0.01).set_opacity(0)

    def fade_id(self) -> Animation:
        """Animate the ID label fading in from invisible."""
        if self.id_label is not None:
            self.id_label.set_opacity(0)
            return self.id_label.animate.set_opacity(1)
        return Wait(0)

    def activate(self, color: str | None = None) -> Animation:
        c = color or self.base_color
        return self.box.animate.set_fill(c, opacity=0.75)

    def deactivate(self) -> Animation:
        return self.box.animate.set_fill(self.base_color, opacity=0.20)

    def set_id(self, new_id: int) -> Animation:
        if self.id_label is None:
            return Wait(0)
        new_lbl = Text(str(new_id), font_size=13, color=self.base_color)
        new_lbl.move_to(self.id_label.get_center())
        return Transform(self.id_label, new_lbl)

    def merge_with(self, other: "TokenBox", result: "TokenBox") -> AnimationGroup:
        result.set_opacity(0)
        mid = midpoint(self.get_center(), other.get_center())
        return AnimationGroup(
            self.animate.move_to(mid).set_opacity(0),
            other.animate.move_to(mid).set_opacity(0),
            result.animate.set_opacity(1),
            lag_ratio=0.25,
        )

    def split_into(self, boxes: list["TokenBox"]) -> AnimationGroup:
        for b in boxes:
            b.set_opacity(0)
            b.move_to(self.get_center())
        return AnimationGroup(
            self.animate.set_opacity(0),
            *[b.animate.set_opacity(1) for b in boxes],
            lag_ratio=0.12,
        )


# ═══════════════════════════════════════════════════════════
# 3.  SpecialTokenBox
# ═══════════════════════════════════════════════════════════

class SpecialTokenBox(TokenBox):
    """
    TokenBox variant for [CLS], [SEP], [PAD] etc.

    Extra animations
    ----------------
    .show_tooltip()   → alias for show_purpose (video script compat)
    .hide_tooltip()   → alias for hide_purpose
    .show_purpose()   → fade in purpose label
    .hide_purpose()   → fade out purpose label
    """

    def __init__(
        self,
        text: str = "[CLS]",
        token_id: int | None = None,
        purpose: str = "",
        **kwargs,
    ):
        kwargs.pop("color", None)
        super().__init__(
            text=text,
            token_id=token_id,
            color=SPECIAL_TOKEN_COLOR,
            font_size=18,
            show_id=True,
            **kwargs,
        )
        self.purpose_str = purpose

        dashed = DashedVMobject(self.box.copy(), num_dashes=18)
        dashed.set_stroke(color=SPECIAL_TOKEN_COLOR, width=1.5, opacity=0.6)
        self.add(dashed)

        self._purpose_label = None
        if purpose:
            self._purpose_label = Text(
                purpose,
                font_size=11,
                color=SPECIAL_TOKEN_COLOR,
                slant=ITALIC,
            ).next_to(self, DOWN, buff=0.30)

    def show_purpose(self) -> Animation:
        if self._purpose_label:
            self.add(self._purpose_label)
            self._purpose_label.set_opacity(0)
            return self._purpose_label.animate.set_opacity(1)
        return Wait(0)

    def hide_purpose(self) -> Animation:
        if self._purpose_label:
            return FadeOut(self._purpose_label)
        return Wait(0)

    # ── aliases used in video scripts ─────────
    def show_tooltip(self) -> Animation:
        return self.show_purpose()

    def hide_tooltip(self) -> Animation:
        return self.hide_purpose()


# ═══════════════════════════════════════════════════════════
# 4.  TokenRow
# ═══════════════════════════════════════════════════════════

class TokenRow(VGroup):
    """
    A horizontal sequence of TokenBoxes.

    Parameters
    ----------
    tokens    : list[str]
    ids/token_ids : list[int] | None   (accepts either kwarg name)
    colors    : list[str] | None
    font_size : int
    buff      : float
    show_ids  : bool
    center    : np.ndarray

    Key animations
    --------------
    .appear(lag)              → AnimationGroup
    .disappear(lag)           → AnimationGroup
    .highlight_token(i)       → Animation
    .activate_token(i)        → Animation
    .deactivate_token(i)      → Animation
    .wave_activate(lag)       → AnimationGroup
    .show_all_ids(scene, delay) → drives scene, reveals each ID label in turn
    .add_token(text, id)      → (TokenBox, Animation)
    .connection_arrows(other) → VGroup of arrows
    """

    def __init__(
        self,
        tokens: list[str] | None = None,
        ids: list[int] | None = None,
        token_ids: list[int] | None = None,   # alias
        colors: list[str] | None = None,
        font_size: int = 22,
        buff: float = 0.18,
        show_ids: bool = True,
        center=ORIGIN,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if tokens is None:
            tokens, ids = make_token_sequence()

        # accept either 'ids' or 'token_ids'
        resolved_ids = ids if ids is not None else token_ids

        self.tokens     = tokens
        self.ids        = resolved_ids or [None] * len(tokens)
        self.buff       = buff
        self.token_boxes: list[TokenBox] = []

        for i, (tok, tid) in enumerate(zip(self.tokens, self.ids)):
            color = (colors[i] if colors and i < len(colors) else None) \
                     or TOKEN_PALETTE[i % len(TOKEN_PALETTE)]
            if tok in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>",
                       "<|endoftext|>", "[MASK]", "<pad>"):
                box = SpecialTokenBox(tok, token_id=tid)
            else:
                box = TokenBox(
                    tok,
                    token_id=tid,
                    color=color,
                    font_size=font_size,
                    show_id=show_ids,
                    is_subword=tok.startswith("##"),
                )
            self.token_boxes.append(box)

        row = VGroup(*self.token_boxes).arrange(RIGHT, buff=buff)
        row.move_to(center)
        self.add(row)

    # ── animations ────────────────────────────

    def appear(self, lag: float = 0.10) -> AnimationGroup:
        return AnimationGroup(
            *[FadeIn(b, shift=UP * 0.2) for b in self.token_boxes],
            lag_ratio=lag,
        )

    def disappear(self, lag: float = 0.08) -> AnimationGroup:
        return AnimationGroup(
            *[FadeOut(b, shift=DOWN * 0.1) for b in self.token_boxes],
            lag_ratio=lag,
        )

    def highlight_token(self, i: int, color: str = HIGHLIGHT_COLOR) -> Animation:
        return self.token_boxes[i].highlight(color)

    def activate_token(self, i: int) -> Animation:
        return self.token_boxes[i].activate()

    def deactivate_token(self, i: int) -> Animation:
        return self.token_boxes[i].deactivate()

    def wave_activate(self, lag: float = 0.12) -> AnimationGroup:
        anims = [
            Succession(b.activate(), Wait(0.1), b.deactivate())
            for b in self.token_boxes
        ]
        return AnimationGroup(*anims, lag_ratio=lag)

    def show_all_ids(self, scene: Scene, delay: float = 0.06) -> None:
        """
        Animate each token's ID label appearing one by one.
        Drives the scene directly — call as tok_row.show_all_ids(self, delay=0.06).

        Tokens whose id_label is None (no ID assigned or show_id=False) are skipped.
        """
        for box in self.token_boxes:
            if box.id_label is not None:
                box.id_label.set_opacity(0)
                scene.play(
                    box.id_label.animate.set_opacity(1),
                    run_time=max(delay * 2, 0.05),
                )
                scene.wait(delay)

    def add_token(
        self,
        text: str,
        token_id: int | None = None,
        color: str = TOKEN_COLOR,
    ) -> tuple["TokenBox", Animation]:
        new_box = TokenBox(text, token_id=token_id, color=color)
        if self.token_boxes:
            new_box.next_to(self.token_boxes[-1], RIGHT, buff=self.buff)
        self.token_boxes.append(new_box)
        self.add(new_box)
        return new_box, FadeIn(new_box, shift=RIGHT * 0.3)

    def connection_arrows(
        self,
        other_row: "TokenRow",
        color: str = DIM_COLOR,
        one_to_one: bool = True,
    ) -> VGroup:
        arrows = VGroup()
        pairs = (
            zip(self.token_boxes, other_row.token_boxes)
            if one_to_one
            else [(self.token_boxes[0], b) for b in other_row.token_boxes]
        )
        for src, tgt in pairs:
            a = Arrow(
                src.get_bottom(), tgt.get_top(),
                buff=0.08, color=color,
                stroke_width=1.5,
                max_tip_length_to_length_ratio=0.2,
            )
            arrows.add(a)
        return arrows


# ═══════════════════════════════════════════════════════════
# 5.  TokenSplitAnimation
# ═══════════════════════════════════════════════════════════

class TokenSplitAnimation(VGroup):
    """
    Shows a single word splitting into BPE sub-word tokens.

    Parameters
    ----------
    word      : str        – full word, e.g. "unhappiness"
                             OR a space-separated sentence (legacy usage)
    subwords  : list[str]  – split result
    text      : str        – alias for `word` (legacy kwarg compat)
    tokens    : list[str]  – alias for `subwords` (legacy kwarg compat)
    ids       : list[int]
    font_size : int

    Key animations (drive scene directly)
    ----------------------------------------
    .show_word(scene)
    .show_split_lines(scene)
    .separate_pieces(scene)
    .show_subword_boxes(scene)
    .play_full(scene)
    .full_sequence(scene)   → alias for play_full; returns self
    """

    def __init__(
        self,
        word: str = "playing",
        subwords: list[str] | None = None,
        text: str | None = None,          # legacy alias
        tokens: list[str] | None = None,  # legacy alias
        ids: list[int] | None = None,
        font_size: int = 36,
        **kwargs,
    ):
        super().__init__(**kwargs)
        actual_word     = text    if text    is not None else word
        actual_subwords = tokens  if tokens  is not None else subwords

        self.word     = actual_word
        self.subwords = actual_subwords or [actual_word[:len(actual_word)//2],
                                             "##" + actual_word[len(actual_word)//2:]]
        self.ids       = ids or list(range(len(self.subwords)))
        self.font_size = font_size

        self.word_text = Text(actual_word, font_size=font_size, color=WHITE)
        self.add(self.word_text)

        self._split_lines: list[Line] = []
        self._subword_boxes: list[TokenBox] = []
        self._build_split_lines()

    # ── internal ──────────────────────────────
    def _char_x_positions(self) -> list[float]:
        char_w = self.word_text.width / max(len(self.word), 1)
        x0 = self.word_text.get_left()[0]
        return [x0 + i * char_w for i in range(len(self.word) + 1)]

    def _build_split_lines(self):
        xs    = self._char_x_positions()
        idx   = 0
        y_top = self.word_text.get_top()[1]    + 0.15
        y_bot = self.word_text.get_bottom()[1] - 0.15
        for sw in self.subwords[:-1]:
            clean = sw.lstrip("#")
            idx  += len(clean)
            if idx < len(xs):
                line = DashedLine(
                    [xs[idx], y_top, 0],
                    [xs[idx], y_bot, 0],
                    color=HIGHLIGHT_COLOR,
                    stroke_width=2,
                    dash_length=0.08,
                )
                line.set_opacity(0)
                self._split_lines.append(line)
                self.add(line)

    # ── step animations ───────────────────────
    def show_word(self, scene: Scene) -> None:
        scene.play(FadeIn(self.word_text, shift=UP * 0.1))
        scene.wait(0.4)

    def show_split_lines(self, scene: Scene) -> None:
        scene.play(AnimationGroup(
            *[line.animate.set_opacity(1) for line in self._split_lines],
            lag_ratio=0.2,
        ))
        scene.wait(0.3)

    def separate_pieces(self, scene: Scene) -> None:
        xs      = self._char_x_positions()
        idx     = 0
        pieces  = []
        spacing = 0.35
        for i, sw in enumerate(self.subwords):
            clean = sw.lstrip("#")
            color = TOKEN_PALETTE[i % len(TOKEN_PALETTE)]
            piece = Text(sw, font_size=self.font_size - 4, color=color)
            x_mid = (xs[idx] + xs[min(idx + len(clean), len(xs) - 1)]) / 2
            piece.move_to([x_mid, self.word_text.get_center()[1], 0])
            pieces.append(piece)
            idx += len(clean)

        total_w = sum(p.width for p in pieces) + spacing * (len(pieces) - 1)
        x_start = -total_w / 2
        targets = []
        cursor  = x_start
        for p in pieces:
            targets.append(np.array([cursor + p.width / 2,
                                      self.word_text.get_center()[1] - 1.0, 0]))
            cursor += p.width + spacing

        self._piece_mobs = pieces
        for p in pieces:
            scene.add(p)

        scene.play(
            FadeOut(self.word_text),
            *[FadeOut(l) for l in self._split_lines],
            *[p.animate.move_to(t) for p, t in zip(pieces, targets)],
            run_time=1.0,
        )
        scene.wait(0.3)

    def show_subword_boxes(self, scene: Scene) -> None:
        if not hasattr(self, "_piece_mobs"):
            return
        for i, (piece, sw, tid) in enumerate(
            zip(self._piece_mobs, self.subwords, self.ids)
        ):
            color = TOKEN_PALETTE[i % len(TOKEN_PALETTE)]
            box   = TokenBox(sw, token_id=tid, color=color,
                             is_subword=sw.startswith("##"), font_size=20)
            box.move_to(piece.get_center() + DOWN * 0.05)
            self._subword_boxes.append(box)
            scene.add(box)
        scene.play(
            *[FadeOut(p) for p in self._piece_mobs],
            *[FadeIn(b)  for b in self._subword_boxes],
        )
        scene.wait(0.5)

    def play_full(self, scene: Scene) -> None:
        self.show_word(scene)
        self.show_split_lines(scene)
        self.separate_pieces(scene)
        self.show_subword_boxes(scene)

    def full_sequence(self, scene: Scene) -> "TokenSplitAnimation":
        """Alias of play_full — returns self so callers can store a ref."""
        self.play_full(scene)
        return self


# ═══════════════════════════════════════════════════════════
# 6.  BPEStepDisplay
# ═══════════════════════════════════════════════════════════

class BPEStepDisplay(VGroup):
    """Step-by-step BPE merge visualisation."""

    def __init__(
        self,
        before_tokens: list[str] | None = None,
        pair_to_merge: tuple[str, str] = ("l", "o"),
        after_tokens: list[str] | None = None,
        step_number: int = 1,
        pair_freq: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.before_tokens = before_tokens or ["l", "o", "w", "e", "r"]
        self.after_tokens  = after_tokens  or ["lo", "w", "e", "r"]
        self.pair          = pair_to_merge
        self.step_number   = step_number
        self.pair_freq     = pair_freq

        self._before_row: TokenRow | None = None
        self._after_row:  TokenRow | None = None

    def show_before(self, scene: Scene) -> None:
        step_lbl = Text(f"BPE Step {self.step_number}", font_size=20,
                        color=HIGHLIGHT_COLOR).to_edge(UP).shift(DOWN * 0.5)
        freq_lbl = Text(
            f'Most frequent pair: "{self.pair[0]}" + "{self.pair[1]}"  (×{self.pair_freq})',
            font_size=18, color=WHITE,
        ).next_to(step_lbl, DOWN, buff=0.2)
        self._before_row = TokenRow(
            tokens=self.before_tokens,
            ids=list(range(len(self.before_tokens))),
            show_ids=False, font_size=24,
        )
        self._before_row.move_to(UP * 0.5)
        scene.play(FadeIn(step_lbl), FadeIn(freq_lbl), self._before_row.appear())
        scene.wait(0.4)

    def highlight_pair(self, scene: Scene) -> None:
        if self._before_row is None:
            return
        names = [b.token_text for b in self._before_row.token_boxes]
        idxs: list[int] = []
        for i, n in enumerate(names):
            if n == self.pair[0] and not idxs:
                idxs.append(i)
            elif idxs and n == self.pair[1] and len(idxs) == 1:
                idxs.append(i)
                break
        anims = [self._before_row.token_boxes[i].highlight(HIGHLIGHT_COLOR)
                 for i in idxs if i < len(self._before_row.token_boxes)]
        pair_group = VGroup(
            *[self._before_row.token_boxes[i]
              for i in idxs if i < len(self._before_row.token_boxes)]
        )
        brace = Brace(pair_group, UP, color=HIGHLIGHT_COLOR)
        merge_lbl = brace.get_text("merge!", font_size=16).set_color(HIGHLIGHT_COLOR)
        scene.play(*anims)
        scene.play(FadeIn(brace), FadeIn(merge_lbl))
        scene.wait(0.4)

    def show_merge_arrow(self, scene: Scene) -> None:
        if self._before_row is None:
            return
        arrow = Arrow(
            self._before_row.get_bottom() + DOWN * 0.1,
            self._before_row.get_bottom() + DOWN * 0.9,
            color=HIGHLIGHT_COLOR, stroke_width=2,
        )
        scene.play(GrowArrow(arrow))
        scene.wait(0.2)

    def show_after(self, scene: Scene) -> None:
        self._after_row = TokenRow(
            tokens=self.after_tokens,
            ids=list(range(len(self.after_tokens))),
            show_ids=False, font_size=24,
        )
        self._after_row.move_to(DOWN * 1.2)
        scene.play(self._after_row.appear())
        scene.wait(0.6)

    def play_full(self, scene: Scene) -> None:
        self.show_before(scene)
        self.highlight_pair(scene)
        self.show_merge_arrow(scene)
        self.show_after(scene)


# ═══════════════════════════════════════════════════════════
# 7.  VocabTable
# ═══════════════════════════════════════════════════════════

class VocabTable(VGroup):
    """Token → ID lookup table with animated row search."""

    def __init__(
        self,
        vocab: dict | None = None,
        max_rows: int = 10,
        font_size: int = 16,
        row_height: float = 0.38,
        col_width: float = 1.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab      = vocab or DEMO_VOCAB
        self.font_size  = font_size
        self.row_height = row_height
        self.col_width  = col_width

        items = list(self.vocab.items())[:max_rows]
        if len(self.vocab) > max_rows:
            items.append(("…", "…"))

        self.rows: list[VGroup] = []
        self._token_to_row: dict[str, int] = {}
        self._build(items)

    def _build(self, items):
        hdr_tok = Text("token",    font_size=self.font_size, color=TOKEN_COLOR, weight=BOLD)
        hdr_id  = Text("token id", font_size=self.font_size, color=TOKEN_COLOR, weight=BOLD)
        hdr_tok.move_to(LEFT * self.col_width / 2)
        hdr_id.move_to(RIGHT * self.col_width / 2)
        header = VGroup(hdr_tok, hdr_id)
        header.add(
            Line(header.get_left(), header.get_right(),
                 color=TOKEN_COLOR, stroke_width=1).next_to(header, DOWN, buff=0.06)
        )
        self.rows.append(header)
        self.add(header)
        for i, (tok, tid) in enumerate(items):
            row = self._make_row(tok, str(tid), i)
            row.next_to(self.rows[-1], DOWN, buff=0.05)
            self.rows.append(row)
            self.add(row)
            if tok != "…":
                self._token_to_row[tok] = i + 1

    def _make_row(self, token, tid, idx):
        bg = Rectangle(
            width=self.col_width * 2 + 0.2, height=self.row_height,
            fill_color="#2A2A3E",
            fill_opacity=0.5 if idx % 2 == 0 else 0.2,
            stroke_width=0,
        )
        tok_lbl = Text(token, font_size=self.font_size, color=WHITE)
        id_lbl  = Text(tid,   font_size=self.font_size, color=EMBEDDING_COLOR)
        tok_lbl.move_to(bg.get_center() + LEFT  * self.col_width / 2)
        id_lbl.move_to( bg.get_center() + RIGHT * self.col_width / 2)
        return VGroup(bg, tok_lbl, id_lbl)

    def appear(self, lag: float = 0.06) -> AnimationGroup:
        return AnimationGroup(
            *[FadeIn(r, shift=RIGHT * 0.1) for r in self.rows],
            lag_ratio=lag,
        )

    def flash_row(self, row_idx: int, color: str = HIGHLIGHT_COLOR) -> Animation:
        if row_idx >= len(self.rows):
            return Wait(0)
        return self.rows[row_idx][0].animate.set_fill(color, opacity=0.6)

    def lookup(self, token: str, scene: Scene) -> int | None:
        row_idx = self._token_to_row.get(token)
        if row_idx is None:
            scene.play(Wiggle(self))
            return None
        for i in range(1, row_idx):
            scene.play(self.rows[i][0].animate.set_fill(EMBEDDING_COLOR, opacity=0.3), run_time=0.07)
            scene.play(self.rows[i][0].animate.set_fill("#2A2A3E", opacity=0.5),       run_time=0.07)
        scene.play(self.flash_row(row_idx, HIGHLIGHT_COLOR))
        found = Text(f'Found "{token}" → ID {self.vocab[token]}',
                     font_size=14, color=HIGHLIGHT_COLOR).next_to(self, DOWN, buff=0.2)
        scene.play(FadeIn(found))
        scene.wait(0.6)
        scene.play(FadeOut(found),
                   self.rows[row_idx][0].animate.set_fill("#2A2A3E", opacity=0.5))
        return row_idx


# ═══════════════════════════════════════════════════════════
# 8.  ContextWindowBar
# ═══════════════════════════════════════════════════════════

class ContextWindowBar(VGroup):
    """
    Horizontal strip of token slots representing the context window.

    Parameters
    ----------
    max_tokens   : int
    filled       : int   – pre-filled slots on creation
    slot_width   : float
    slot_height  : float
    show_numbers : bool

    Key animations  (scene-driving)
    --------------------------------
    .appear(scene)               → FadeIn the whole bar
    .fill_slot(i, scene)         → animate one slot being filled
    .fill_up_to(n, scene)        → fill slots 0..n-1
    .fill_next(scene)            → fill the next empty slot
    .fill_all(lag, scene)        → fill all slots
    .overflow_flash(scene)       → red flash at context limit
    .show_usage_label()          → returns Text mob (add + FadeIn yourself)
    .show_position_numbers(scene)
    .truncate_to(n, scene)
    """

    FILLED_COLOR   = TOKEN_COLOR
    EMPTY_COLOR    = DIM_COLOR
    OVERFLOW_COLOR = OUTPUT_COLOR

    def __init__(
        self,
        max_tokens: int = 16,
        filled: int = 0,
        slot_width: float = 0.38,
        slot_height: float = 0.38,
        show_numbers: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_tokens   = max_tokens
        self.filled       = filled
        self.slot_width   = slot_width
        self.slot_height  = slot_height
        self.show_numbers = show_numbers
        self._next_slot   = filled

        self.slots: list[Rectangle] = []
        self._pos_labels: list[Text] = []
        self._build()

    def _make_slot(self, is_filled: bool) -> Rectangle:
        color = self.FILLED_COLOR if is_filled else self.EMPTY_COLOR
        return Rectangle(
            width=self.slot_width, height=self.slot_height,
            fill_color=color,
            fill_opacity=0.7 if is_filled else 0.15,
            stroke_color=color, stroke_width=1,
        )

    def _build(self):
        for i in range(self.max_tokens):
            slot = self._make_slot(i < self.filled)
            self.slots.append(slot)
        row = VGroup(*self.slots).arrange(RIGHT, buff=0.04)
        row.move_to(ORIGIN)
        self.add(row)

        max_lbl = Text(f"max = {self.max_tokens} tokens",
                       font_size=13, color=DIM_COLOR)
        max_lbl.next_to(row, RIGHT, buff=0.2)
        self.add(max_lbl)

        if self.show_numbers:
            for i, slot in enumerate(self.slots):
                lbl = Text(str(i), font_size=9, color=DIM_COLOR)
                lbl.next_to(slot, DOWN, buff=0.04)
                self._pos_labels.append(lbl)
                self.add(lbl)

    # ── animations ────────────────────────────
    def appear(self, scene: Scene | None = None) -> Animation:
        anim = FadeIn(self, shift=UP * 0.1)
        if scene is not None:
            scene.play(anim)
        return anim

    def fill_slot(self, i: int, scene: Scene, color: str | None = None) -> None:
        c = color or self.FILLED_COLOR
        scene.play(
            self.slots[i].animate.set_fill(c, opacity=0.7).set_stroke(c),
            run_time=0.2,
        )

    def fill_up_to(self, n: int, scene: Scene, lag: float = 0.06) -> None:
        n = min(n, self.max_tokens)
        anims = [
            self.slots[i].animate.set_fill(self.FILLED_COLOR, opacity=0.7)
                                  .set_stroke(self.FILLED_COLOR)
            for i in range(n)
        ]
        scene.play(AnimationGroup(*anims, lag_ratio=lag))
        self._next_slot = n

    def fill_next(self, scene: Scene, color: str | None = None) -> None:
        if self._next_slot >= self.max_tokens:
            self.overflow_flash(scene)
            return
        self.fill_slot(self._next_slot, scene, color)
        self._next_slot += 1

    def fill_all(self, lag: float = 0.08, scene: Scene | None = None) -> AnimationGroup:
        anims = [
            self.slots[i].animate.set_fill(self.FILLED_COLOR, opacity=0.7)
                                  .set_stroke(self.FILLED_COLOR)
            for i in range(self.max_tokens)
        ]
        anim = AnimationGroup(*anims, lag_ratio=lag)
        if scene:
            scene.play(anim)
        return anim

    def show_usage_label(self) -> Text:
        used = self._next_slot
        lbl = Text(
            f"{used} / {self.max_tokens} tokens used",
            font_size=16, color=TOKEN_COLOR,
        )
        lbl.next_to(self, DOWN, buff=0.25)
        return lbl

    def overflow_flash(self, scene: Scene) -> None:
        scene.play(AnimationGroup(
            *[slot.animate.set_fill(self.OVERFLOW_COLOR, opacity=0.9)
              for slot in self.slots],
            lag_ratio=0.02,
        ))
        warning = Text("⚠  Context limit reached!", font_size=18,
                       color=self.OVERFLOW_COLOR).next_to(self, DOWN, buff=0.25)
        scene.play(FadeIn(warning))
        scene.wait(0.8)
        scene.play(
            FadeOut(warning),
            *[slot.animate.set_fill(self.FILLED_COLOR, opacity=0.7)
              for slot in self.slots],
        )

    def show_position_numbers(self, scene: Scene) -> None:
        if not self._pos_labels:
            return
        scene.play(AnimationGroup(
            *[FadeIn(l) for l in self._pos_labels], lag_ratio=0.03,
        ))

    def truncate_to(self, n: int, scene: Scene) -> None:
        if n >= self.max_tokens:
            return
        scene.play(AnimationGroup(
            *[self.slots[i].animate.set_fill(self.EMPTY_COLOR, opacity=0.15)
                                    .set_stroke(self.EMPTY_COLOR)
              for i in range(n, self.max_tokens)],
            lag_ratio=0.04,
        ))
        self._next_slot = n


# ═══════════════════════════════════════════════════════════
# 9.  TokenIDMapping
# ═══════════════════════════════════════════════════════════

class TokenIDMapping(VGroup):
    """
    Shows token → ID connections.

    Mode A — pass a TokenRow:
        mapping = TokenIDMapping(token_row=row)

    Mode B — pass (token, id) pairs (used in video script):
        mapping = TokenIDMapping("the", 1, color="#4CAF50")

    Parameters
    ----------
    token_row  : TokenRow | None
    id_color   : str
    arc_angle  : float
    token_text : str   – single token string (Mode B)
    token_id   : int   (Mode B)
    color      : str   – alias for id_color in Mode B

    Key animations
    --------------
    .reveal(lag)          → AnimationGroup
    .highlight_pair(i)    → AnimationGroup
    .flash_all(lag)       → AnimationGroup
    .appear(scene)        → drives scene directly (Mode B)
    """

    def __init__(
        self,
        token_text: str | None = None,
        token_id: int | None = None,
        token_row: "TokenRow | None" = None,
        id_color: str = EMBEDDING_COLOR,
        arc_angle: float = -TAU / 8,
        color: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.id_color = color or id_color
        self.arcs: list[CurvedArrow] = []
        self.id_labels: list[Text]   = []
        self._token_boxes: list[TokenBox] = []

        if token_row is not None:
            self.token_row = token_row
            self._build_from_row(token_row, arc_angle)
        elif token_text is not None:
            self.token_row = None
            self._build_standalone(token_text, token_id, arc_angle)

    def _build_from_row(self, row: "TokenRow", arc_angle: float):
        for box in row.token_boxes:
            tid = box.token_id
            if tid is None:
                continue
            lbl = Text(str(tid), font_size=16, color=self.id_color)
            lbl.next_to(box, DOWN, buff=1.0)
            arc = CurvedArrow(
                box.get_bottom() + DOWN * 0.05,
                lbl.get_top()    + UP   * 0.05,
                angle=arc_angle,
                color=self.id_color,
                stroke_width=1.5, stroke_opacity=0.7,
            )
            self.id_labels.append(lbl)
            self.arcs.append(arc)
            self._token_boxes.append(box)
            self.add(arc, lbl)

    def _build_standalone(self, token_text: str, token_id: int | None, arc_angle: float):
        box = TokenBox(
            token_text, token_id=token_id,
            color=self.id_color, font_size=22, show_id=False,
        )
        self._token_boxes.append(box)
        self.add(box)

        if token_id is not None:
            lbl = Text(str(token_id), font_size=18, color=self.id_color)
            lbl.next_to(box, DOWN, buff=0.8)
            arc = CurvedArrow(
                box.get_bottom() + DOWN * 0.05,
                lbl.get_top()    + UP   * 0.05,
                angle=arc_angle,
                color=self.id_color,
                stroke_width=1.5, stroke_opacity=0.7,
            )
            self.id_labels.append(lbl)
            self.arcs.append(arc)
            self.add(arc, lbl)

    # ── animations ────────────────────────────
    def reveal(self, lag: float = 0.10) -> AnimationGroup:
        pairs = [
            AnimationGroup(Create(arc), FadeIn(lbl), lag_ratio=0.3)
            for arc, lbl in zip(self.arcs, self.id_labels)
        ]
        return AnimationGroup(*pairs, lag_ratio=lag)

    def highlight_pair(self, i: int) -> AnimationGroup:
        if i >= len(self.arcs):
            return Wait(0)
        return AnimationGroup(
            Indicate(self._token_boxes[i], color=HIGHLIGHT_COLOR),
            self.arcs[i].animate.set_stroke(HIGHLIGHT_COLOR, width=3, opacity=1.0),
            Indicate(self.id_labels[i], color=HIGHLIGHT_COLOR),
            lag_ratio=0.15,
        )

    def flash_all(self, lag: float = 0.12) -> AnimationGroup:
        return AnimationGroup(
            *[self.highlight_pair(i) for i in range(len(self.arcs))],
            lag_ratio=lag,
        )

    def appear(self, scene: Scene) -> None:
        """Mode B convenience — drives scene directly."""
        scene.play(FadeIn(self))
        if self.arcs:
            scene.play(self.reveal())


# ─────────────────────────────────────────────────────────────────────────────
# Demo Scene
# manim -pql token_box.py TokenizationDemoScene
# ─────────────────────────────────────────────────────────────────────────────

class TokenizationDemoScene(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        title = Text("Tokenization Assets", font_size=32, color=WHITE, weight=BOLD)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))

        # 1. RawTextDisplay
        raw = RawTextDisplay("The transformer is all you need", font_size=28)
        raw.shift(UP * 1.5)
        self.add(raw)
        raw.reveal_chars(self, delay=0.025)
        self.play(raw.highlight_word(1))
        self.play(raw.dim_all())
        self.play(raw.undim_all())
        self.play(raw.fade_to_tokens())

        # 2. TokenRow + show_all_ids
        row = TokenRow(
            tokens=["[CLS]", "the", "cat", "sat", "[SEP]"],
            ids=[101, 1996, 4937, 2938, 102],
            show_ids=False,   # start hidden so show_all_ids can reveal them
        )
        self.play(row.appear())
        row.show_all_ids(self, delay=0.06)   # ← new method
        self.play(row.wave_activate())
        self.wait(0.5)
        self.play(FadeOut(row))

        # 3. ContextWindowBar
        ctx = ContextWindowBar(max_tokens=24, filled=0, show_numbers=True)
        ctx.shift(DOWN * 0.3)
        ctx.appear(self)
        ctx.fill_up_to(18, self)
        usage = ctx.show_usage_label()
        self.play(FadeIn(usage))
        self.wait(0.5)
        for _ in range(7):
            ctx.fill_next(self)
        self.play(FadeOut(ctx), FadeOut(usage))

        # 4. SpecialTokenBox tooltips
        specials = VGroup(
            SpecialTokenBox("[CLS]",  token_id=101, purpose="classification"),
            SpecialTokenBox("[SEP]",  token_id=102, purpose="separator"),
            SpecialTokenBox("[PAD]",  token_id=0,   purpose="padding"),
            SpecialTokenBox("[MASK]", token_id=103, purpose="masked LM"),
        ).arrange(RIGHT, buff=0.5)
        self.play(FadeIn(specials))
        for sp in specials:
            self.play(sp.show_tooltip(), run_time=0.3)
            self.wait(0.4)
            self.play(sp.hide_tooltip(), run_time=0.2)
        self.play(FadeOut(specials))

        # 5. TokenIDMapping standalone (Mode B)
        mappings = VGroup(*[
            TokenIDMapping(tok, tid, color=_token_color(i))
            for i, (tok, tid) in enumerate([
                ("the", 1), ("cat", 2), ("sat", 3), ("[CLS]", 101),
            ])
        ]).arrange(DOWN, buff=0.5)
        for m in mappings:
            m.appear(self)
            self.wait(0.1)
        self.wait(0.8)
        self.play(FadeOut(mappings))

        fin = Text("token_box.py  ✓", font_size=36, color=TOKEN_COLOR, weight=BOLD)
        self.play(Write(fin))
        self.wait(1.5)