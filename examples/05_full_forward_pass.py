"""
VIDEO 5 — "Full Transformer Forward Pass"
==========================================
Runtime: ~5 minutes
Covers:
  - End-to-end: text in → token out
  - All components in one flowing animation
  - Token generation loop (autoregressive)
  - KV cache intuition

Run:
    manim -pqh 05_full_forward_pass.py FullForwardPass
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import numpy as np
from base.shapes import (
    MatrixBox, VectorBar, LabeledBlock, SoftmaxCurve, MathLabel,
    TOKEN_COLOR, EMBEDDING_COLOR, ATTENTION_COLOR,
    FFN_COLOR, NORM_COLOR, OUTPUT_COLOR,KEY_COLOR,
    HIGHLIGHT_COLOR, DIM_COLOR, BACKGROUND_COLOR, 
)
from base.animations import (
    data_flow_arrow, forward_pass_pulse, layer_stack_anim,
    highlight_sequence, pulse_glow, zoom_into,
    token_stream_anim, equation_reveal,
)
from base.utils import (
    apply_dark_theme, make_sample_logits,
    make_token_sequence, softmax, step_log,
)
from tokenization.token_box import (
    TokenRow, TokenBox, SpecialTokenBox, _token_color,
)


class FullForwardPass(Scene):
    def construct(self):
        apply_dark_theme(self)
        step_log("Starting FullForwardPass", "FullForwardPass")

        # ══════════════════════════════════════════
        # SCENE 1 — Title
        # ══════════════════════════════════════════
        title = Text("Full Transformer Forward Pass", font_size=40, color=WHITE, weight=BOLD)
        sub   = Text("From text to next token — step by step", font_size=20, color=DIM_COLOR
                     ).next_to(title, DOWN, buff=0.3)
        self.play(Write(title))
        self.play(FadeIn(sub))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(sub))

        # ══════════════════════════════════════════
        # SCENE 2 — Architecture overview (all blocks)
        # ══════════════════════════════════════════
        step_log("Building architecture overview", "FullForwardPass")

        overview_lbl = Text("Architecture at a glance", font_size=24,
                            color=WHITE, weight=BOLD).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(overview_lbl))

        components = [
            ("Input Text",          TOKEN_COLOR),
            ("Tokenisation",        TOKEN_COLOR),
            ("Embedding + PE",      EMBEDDING_COLOR),
            ("Transformer Block ×N",ATTENTION_COLOR),
            ("Layer Norm",          NORM_COLOR),
            ("Output Logits",       OUTPUT_COLOR),
            ("Softmax + Sample",    FFN_COLOR),
            ("Next Token",          HIGHLIGHT_COLOR),
        ]
        blocks = [
            LabeledBlock(name, width=2.8, height=0.62, color=color)
            for name, color in components
        ]
        stack = VGroup(*blocks).arrange(DOWN, buff=0.18).scale(0.85)
        stack.shift(DOWN * 0.2)
        self.play(AnimationGroup(*[FadeIn(b, shift=RIGHT * 0.1) for b in blocks], lag_ratio=0.08))
        self.wait(0.5)

        # pulse through all blocks (forward pass)
        self.play(forward_pass_pulse(blocks, pulse_color=HIGHLIGHT_COLOR, lag=0.1))
        self.wait(0.8)
        self.play(*[FadeOut(m) for m in [*blocks, overview_lbl]])

        # ══════════════════════════════════════════
        # SCENE 3 — Step-by-step with zooming
        # ══════════════════════════════════════════

        # ── Step A: Input text ────────────────────
        step_log("Input text scene", "FullForwardPass")
        self._step_banner(self, "Step 1  ·  Input Text", TOKEN_COLOR)

        prompt = Text(
            '"The transformer learns by"',
            font_size=30, color=WHITE,
        ).shift(UP * 0.3)
        self.play(Write(prompt), run_time=0.8)
        self.wait(0.7)
        self.play(FadeOut(prompt))

        # ── Step B: Tokenisation ─────────────────
        self._step_banner(self, "Step 2  ·  Tokenisation", TOKEN_COLOR)

        words     = ["The", "transformer", "learns", "by"]
        ids       = [1, 6, 7, 8]
        tok_row   = TokenRow(words, token_ids=ids, show_ids=True)
        tok_row.shift(UP * 0.5)
        self.play(tok_row.appear())
        tok_row.show_all_ids(self, delay=0.06)
        self.wait(0.5)

        # ── Step C: Embedding + PE ────────────────
        self._step_banner(self, "Step 3  ·  Embedding + Positional Encoding", EMBEDDING_COLOR)

        emb_mat = MatrixBox(rows=4, cols=6, cell_size=0.35, color=EMBEDDING_COLOR)
        emb_mat.shift(LEFT * 3.5)
        emb_lbl = Text("E", font_size=14, color=EMBEDDING_COLOR).next_to(emb_mat, UP, buff=0.1)
        self.play(emb_mat.fill_anim(), FadeIn(emb_lbl))

        # output embeddings
        vec_list = [VectorBar(dim=6, label=w, color=_token_color(i)) for i, w in enumerate(words)]
        emb_vecs = VGroup(*vec_list)
        emb_vecs.arrange(RIGHT, buff=0.25)   # arrange AFTER adding to VGroup
        emb_vecs.shift(RIGHT * 1.0)

        a1, an1 = data_flow_arrow(emb_mat, emb_vecs, label="lookup", color=EMBEDDING_COLOR)
        self.add(a1); self.play(an1)
        self.play(FadeIn(emb_vecs))

        # PE addition
        pe_note = Text("+ Positional Encoding", font_size=17, color=NORM_COLOR
                       ).next_to(emb_vecs, DOWN, buff=0.3)
        self.play(FadeIn(pe_note))
        self.play(
            AnimationGroup(*[v.pulse(color=NORM_COLOR) for v in emb_vecs], lag_ratio=0.1)
        )
        self.wait(0.5)
        self.play(*[FadeOut(m) for m in [tok_row, emb_mat, emb_lbl, emb_vecs, a1, pe_note]])

        # ── Step D: Transformer Block ─────────────
        self._step_banner(self, "Step 4  ·  Transformer Block (×N)", ATTENTION_COLOR)

        # mini block diagram
        blk_parts = [
            ("Layer Norm",          NORM_COLOR,      LEFT * 2.5 + UP * 1.5),
            ("Multi-Head Attention", ATTENTION_COLOR, LEFT * 2.5 + UP * 0.4),
            ("Add & Norm",          NORM_COLOR,      LEFT * 2.5 + DOWN * 0.5),
            ("Feed-Forward (FFN)",  FFN_COLOR,       LEFT * 2.5 + DOWN * 1.5),
            ("Add & Norm",          NORM_COLOR,      LEFT * 2.5 + DOWN * 2.4),
        ]
        blk_mobs = []
        for name, color, shift in blk_parts:
            b = LabeledBlock(name, width=2.8, height=0.58, color=color)
            b.shift(shift)
            blk_mobs.append(b)

        self.play(AnimationGroup(*[FadeIn(b) for b in blk_mobs], lag_ratio=0.1))

        # residual connection (curved line on the right)
        res_start = blk_mobs[0].get_right() + RIGHT * 0.1
        res_end   = blk_mobs[2].get_right() + RIGHT * 0.1
        residual  = CurvedArrow(
            res_start, res_end,
            angle=-TAU / 5,
            color=HIGHLIGHT_COLOR,
            stroke_width=2,
        )
        res_lbl = Text("Residual", font_size=12, color=HIGHLIGHT_COLOR
                       ).next_to(residual, RIGHT, buff=0.05)
        self.play(Create(residual), FadeIn(res_lbl))

        # N-layers annotation
        n_badge = Text("× N layers", font_size=20, color=ATTENTION_COLOR, weight=BOLD)
        n_badge.shift(RIGHT * 2.5)
        self.play(FadeIn(n_badge))

        # forward pass pulse through block
        self.play(forward_pass_pulse(blk_mobs, pulse_color=HIGHLIGHT_COLOR, lag=0.15))
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in blk_mobs + [residual, res_lbl, n_badge]])

        # ── Step E: Output head + Softmax ─────────
        self._step_banner(self, "Step 5  ·  Output Logits + Sampling", OUTPUT_COLOR)

        logits, labels = make_sample_logits(vocab_size=8, top_token="attention", seed=3)
        sc = SoftmaxCurve(
            logits=logits,
            labels=labels,
            temperature=1.0,
            color=OUTPUT_COLOR,
        ).shift(LEFT * 1.0 + DOWN * 0.3)
        sc_lbl = Text("Output distribution", font_size=16, color=OUTPUT_COLOR
                      ).next_to(sc, UP, buff=0.3)
        self.play(FadeIn(sc), FadeIn(sc_lbl))
        self.play(sc.show_values())
        self.wait(0.5)

        # sample a token
        sampled = Text("attention", font_size=28, color=HIGHLIGHT_COLOR, weight=BOLD)
        sampled.shift(RIGHT * 3.5)
        sample_arrow, sample_anim = data_flow_arrow(sc, sampled, label="sample", color=HIGHLIGHT_COLOR)
        self.add(sample_arrow)
        self.play(sample_anim)
        self.play(FadeIn(sampled))
        self.play(pulse_glow(sampled, color=HIGHLIGHT_COLOR))
        self.wait(0.8)
        self.play(*[FadeOut(m) for m in [sc, sc_lbl, sample_arrow, sampled]])

        # ══════════════════════════════════════════
        # SCENE 4 — Autoregressive generation loop
        # ══════════════════════════════════════════
        step_log("Autoregressive loop scene", "FullForwardPass")
        self._step_banner(self, "Autoregressive Generation Loop", HIGHLIGHT_COLOR)

        generated = ["The", "transformer", "learns", "by", "attention"]
        rows = []
        for i, word in enumerate(generated):
            row_words = generated[:i+1]
            row = TokenRow(
                row_words,
                token_ids=list(range(len(row_words))),
                show_ids=False,
            )
            row.scale(0.75)
            rows.append(row)

        current_row = rows[0].copy().shift(UP * 0.5)
        self.play(current_row.appear())

        for i in range(1, len(generated)):
            next_word = generated[i]
            new_tok = TokenBox(next_word, color=_token_color(i))
            new_tok.scale(0.75)
            new_tok.next_to(current_row, RIGHT, buff=0.2)
            new_tok.shift(RIGHT * 1.5).set_opacity(0)
            self.add(new_tok)

            # mini forward pass indicator
            fwd_lbl = Text("forward pass →", font_size=14, color=DIM_COLOR
                           ).next_to(current_row, DOWN, buff=0.3)
            self.play(FadeIn(fwd_lbl), run_time=0.2)
            self.play(
                new_tok.animate.shift(LEFT * 1.5).set_opacity(1),
                run_time=0.4,
            )
            self.play(FadeOut(fwd_lbl), run_time=0.2)

            # rebuild row including new token
            new_row = rows[i].copy().shift(UP * 0.5)
            self.play(
                Transform(current_row, new_row),
                run_time=0.4,
            )
            self.wait(0.2)

        self.wait(0.8)

        loop_note = Text(
            "Each new token is fed back as input for the next prediction",
            font_size=17, color=DIM_COLOR,
        ).to_edge(DOWN).shift(UP * 0.2)
        self.play(FadeIn(loop_note))
        self.wait(1.2)
        self.play(FadeOut(current_row), FadeOut(loop_note))

        # ══════════════════════════════════════════
        # SCENE 5 — KV Cache
        # ══════════════════════════════════════════
        self._step_banner(self, "KV Cache — Why we don't recompute everything", NORM_COLOR)

        kv_note = Text(
            "Without cache: recompute ALL K,V for every new token  →  O(n²) time",
            font_size=19, color=HIGHLIGHT_COLOR,
        ).shift(UP * 1.5)
        self.play(FadeIn(kv_note))

        # cache diagram
        cache_blocks = VGroup(*[
            Rectangle(
                width=0.55, height=0.42,
                fill_color=KEY_COLOR,
                fill_opacity=0.55,
                stroke_color=KEY_COLOR, stroke_width=1,
            )
            for _ in range(5)
        ]).arrange(RIGHT, buff=0.08).shift(UP * 0.2)
        cache_lbl = Text("Cached  K, V", font_size=15, color=KEY_COLOR
                         ).next_to(cache_blocks, UP, buff=0.15)
        self.play(FadeIn(cache_blocks), FadeIn(cache_lbl))

        new_kv = Rectangle(
            width=0.55, height=0.42,
            fill_color=HIGHLIGHT_COLOR,
            fill_opacity=0.75,
            stroke_color=HIGHLIGHT_COLOR, stroke_width=1.5,
        ).next_to(cache_blocks, RIGHT, buff=0.08)
        new_lbl = Text("New\nK,V only", font_size=12, color=HIGHLIGHT_COLOR
                       ).next_to(new_kv, DOWN, buff=0.1)
        self.play(FadeIn(new_kv), FadeIn(new_lbl))

        cache_benefit = Text(
            "With KV cache: only compute K,V for NEW token  →  O(n) time",
            font_size=19, color=TOKEN_COLOR,
        ).shift(DOWN * 1.2)
        self.play(FadeIn(cache_benefit))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ══════════════════════════════════════════
        # SCENE 6 — Final summary
        # ══════════════════════════════════════════
        summary_items = [
            "1. Tokenise input text → integer IDs",
            "2. Embed IDs → dense vectors + positional encoding",
            "3. Pass through N Transformer blocks",
            "   (LayerNorm → MHA → Add → LayerNorm → FFN → Add)",
            "4. Project to vocabulary logits",
            "5. Softmax + sample → next token",
            "6. Append new token, repeat from step 3",
        ]
        summary = VGroup(*[
            Text(t, font_size=19, color=WHITE if not t.startswith(" ") else DIM_COLOR)
            for t in summary_items
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.28).shift(RIGHT * 0.2)
        stitle = Text("Full Pipeline Summary", font_size=26, color=HIGHLIGHT_COLOR, weight=BOLD
                      ).next_to(summary, UP, buff=0.4)
        self.play(FadeIn(stitle))
        for item in summary:
            self.play(FadeIn(item, shift=RIGHT * 0.1), run_time=0.3)
            self.wait(0.2)
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in self.mobjects])
        end = Text("You now understand the Transformer.", font_size=30, color=HIGHLIGHT_COLOR)
        self.play(Write(end))
        self.wait(2)

    # ─────────────────────────────────────────────
    # Helper: step banner
    # ─────────────────────────────────────────────
    @staticmethod
    def _step_banner(scene: Scene, text: str, color: str) -> None:
        """Flash a step label at the top of the screen."""
        banner = Text(text, font_size=20, color=color, weight=BOLD)
        banner.to_edge(UP).shift(DOWN * 0.2)
        scene.play(FadeIn(banner, shift=DOWN * 0.1), run_time=0.35)
        scene.wait(0.2)
        scene.play(FadeOut(banner), run_time=0.25)