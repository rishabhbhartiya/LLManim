"""
VIDEO 1 — "What is a Token?"
==============================
Runtime: ~3–4 minutes
Covers:
  - Raw text input
  - What tokenisation is and why we need it
  - Words → tokens (not always the same!)
  - Subword tokens (##ing, un##happy)
  - Special tokens [CLS] [SEP] [PAD]
  - Token IDs (integer lookup)
  - Context window as a slot bar

Run:
    manim -pqh 01_what_is_a_token.py WhatIsAToken
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
from llmanim.base.shapes import (
    LabeledBlock, MathLabel,
    TOKEN_COLOR, EMBEDDING_COLOR, HIGHLIGHT_COLOR,
    DIM_COLOR, BACKGROUND_COLOR, ATTENTION_COLOR,
)
from llmanim.base.animations import (
    highlight_sequence, fade_label,
    data_flow_arrow, equation_reveal,
)
from llmanim.base.utils import apply_dark_theme, make_token_sequence
from llmanim.tokenization.token_box import (
    RawTextDisplay, TokenBox, SpecialTokenBox,
    TokenRow, TokenSplitAnimation,
    VocabTable, ContextWindowBar, TokenIDMapping,
)

# Color constants (defined here since token_box uses different names)
SPECIAL_TOKEN_COLOR = "#90A4AE"
SUBWORD_COLOR       = "#FF7043"
TOKEN_PALETTE = [
    "#4CAF50","#2196F3","#E91E63","#FF9800",
    "#9C27B0","#00BCD4","#FFEB3B","#F44336",
]
def _token_color(i): return TOKEN_PALETTE[i % len(TOKEN_PALETTE)]


class WhatIsAToken(Scene):
    def construct(self):
        apply_dark_theme(self)

        # ══════════════════════════════════════════
        # SCENE 1 — Title card
        # ══════════════════════════════════════════
        title = Text("What is a Token?", font_size=52, color=WHITE, weight=BOLD)
        sub   = Text(
            "How LLMs read text", font_size=24, color=DIM_COLOR
        ).next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub, shift=UP * 0.1))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(sub))

        # ══════════════════════════════════════════
        # SCENE 2 — LLMs don't read words, they read tokens
        # ══════════════════════════════════════════
        narration = Text(
            "LLMs don't read words.\nThey read tokens.",
            font_size=34, color=WHITE,
            line_spacing=1.4,
        ).shift(UP * 0.5)
        self.play(Write(narration), run_time=1.0)
        self.wait(1.2)
        self.play(FadeOut(narration))

        # ══════════════════════════════════════════
        # SCENE 3 — Show raw text
        # ══════════════════════════════════════════
        label = Text("Input text:", font_size=20, color=DIM_COLOR).to_edge(UP).shift(DOWN * 0.2)
        self.play(FadeIn(label))

        rtd = RawTextDisplay(
            "unhappiness is not the same as happiness",
            font_size=30,
        ).shift(UP * 0.3)
        self.add(rtd)
        rtd.reveal_chars(self, delay=0.025)
        self.wait(0.8)

        # ══════════════════════════════════════════
        # SCENE 4 — Tokenise it  (word-level first)
        # ══════════════════════════════════════════
        question = Text(
            "How would YOU split this into pieces?",
            font_size=22, color=HIGHLIGHT_COLOR,
        ).to_edge(DOWN).shift(UP * 0.3)
        self.play(FadeIn(question))
        self.wait(1.0)
        self.play(FadeOut(question), rtd.fade_to_tokens())
        self.play(FadeOut(label))

        # word-level split first
        word_tokens  = ["un", "happiness", "is", "not", "the", "same", "as", "happiness"]
        word_tsa     = TokenSplitAnimation(
            text="un happiness is not the same as happiness",
            tokens=word_tokens,
        )
        word_row = word_tsa.full_sequence(self)
        self.wait(0.5)

        note = Text(
            "Simple word split — but 'un' and 'happiness' share meaning!",
            font_size=18, color=DIM_COLOR,
        ).to_edge(DOWN).shift(UP * 0.2)
        self.play(FadeIn(note))
        self.wait(1.2)
        self.play(FadeOut(note), FadeOut(word_row))

        # ══════════════════════════════════════════
        # SCENE 5 — Subword tokenisation
        # ══════════════════════════════════════════
        sub_label = Text("Subword tokens (WordPiece / BPE):", font_size=20, color=DIM_COLOR
                         ).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(sub_label))

        sub_tokens = ["un", "##happiness", "is", "not", "the", "same", "as", "##happiness"]
        sub_ids    = [100, 101, 8, 101, 1, 102, 103, 101]
        sub_colors = [
            SUBWORD_COLOR if t.startswith("##") else _token_color(i)
            for i, t in enumerate(sub_tokens)
        ]
        sub_row = TokenRow(sub_tokens, token_ids=sub_ids, colors=sub_colors)
        sub_row.shift(UP * 0.3)
        self.play(sub_row.appear())
        self.wait(0.5)

        # highlight subword pieces
        self.play(sub_row.highlight_token(1, SUBWORD_COLOR))
        self.play(sub_row.highlight_token(7, SUBWORD_COLOR))

        note2 = Text(
            '"##" means this piece continues from the previous token',
            font_size=17, color=SUBWORD_COLOR,
        ).to_edge(DOWN).shift(UP * 0.2)
        self.play(FadeIn(note2))
        self.wait(1.5)
        self.play(FadeOut(note2), FadeOut(sub_label))

        # ══════════════════════════════════════════
        # SCENE 6 — Special tokens
        # ══════════════════════════════════════════
        spec_label = Text("Special tokens:", font_size=20, color=DIM_COLOR
                          ).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(spec_label))

        specials = [
            SpecialTokenBox("[CLS]",  token_id=101),
            SpecialTokenBox("[SEP]",  token_id=102),
            SpecialTokenBox("[PAD]",  token_id=0),
            SpecialTokenBox("[MASK]", token_id=103),
        ]
        spec_row = VGroup(*specials).arrange(RIGHT, buff=0.6).shift(DOWN * 0.3)
        self.play(FadeIn(spec_row))
        self.wait(0.4)

        # show tooltips one by one
        for sp in specials:
            self.play(sp.show_tooltip(), run_time=0.3)
            self.wait(0.5)
            self.play(sp.hide_tooltip(), run_time=0.2)

        self.play(FadeOut(spec_row), FadeOut(spec_label), FadeOut(sub_row))

        # ══════════════════════════════════════════
        # SCENE 7 — Token → Integer ID
        # ══════════════════════════════════════════
        id_label = Text("Every token maps to an integer ID:", font_size=20, color=DIM_COLOR
                        ).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(id_label))

        mappings = VGroup(*[
            TokenIDMapping(tok, tid, color=_token_color(i))
            for i, (tok, tid) in enumerate([
                ("the",    1),
                ("cat",    2),
                ("sat",    3),
                ("[CLS]",  101),
            ])
        ]).arrange(DOWN, buff=0.35).shift(DOWN * 0.2)

        for m in mappings:
            m.appear(self)
            self.wait(0.1)

        self.wait(0.8)
        self.play(FadeOut(mappings), FadeOut(id_label))

        # ══════════════════════════════════════════
        # SCENE 8 — Context window
        # ══════════════════════════════════════════
        ctx_label = Text("Context window = max tokens the model can see:", font_size=20, color=DIM_COLOR
                         ).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(ctx_label))

        ctx = ContextWindowBar(max_tokens=24, filled=0, show_numbers=True)
        ctx.shift(DOWN * 0.3)
        self.play(ctx.appear())
        ctx.fill_up_to(18, self)
        usage = ctx.show_usage_label()
        self.play(FadeIn(usage))
        self.wait(0.8)

        # try to overfill
        for _ in range(7):
            ctx.fill_next(self)
        self.wait(0.5)
        self.play(FadeOut(ctx), FadeOut(usage), FadeOut(ctx_label))

        # ══════════════════════════════════════════
        # SCENE 9 — Summary
        # ══════════════════════════════════════════
        summary_items = [
            "• Text  →  tokens  →  integer IDs",
            "• Tokens ≠ words  (subword splitting)",
            "• Special tokens guide the model",
            "• Context window limits token count",
        ]
        summary = VGroup(*[
            Text(t, font_size=22, color=WHITE) for t in summary_items
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(RIGHT * 0.5)

        stitle = Text("Summary", font_size=30, color=TOKEN_COLOR, weight=BOLD
                      ).next_to(summary, UP, buff=0.4)
        self.play(FadeIn(stitle))
        for item in summary:
            self.play(FadeIn(item, shift=RIGHT * 0.15), run_time=0.4)
            self.wait(0.3)
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])
        end = Text("Next → Token Embeddings", font_size=28, color=EMBEDDING_COLOR)
        self.play(Write(end))
        self.wait(2)