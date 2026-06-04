"""
VIDEO 4 — "Multi-Head Attention"
==================================
Runtime: ~4 minutes
Covers:
  - Why one attention head isn't enough
  - Splitting d_model into h heads
  - Each head learns different patterns
  - Concatenate → project with W_O
  - Head comparison (different heatmaps)

Run:
    manim -pqh 04_multi_head_attention.py MultiHeadAttention
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import numpy as np
from llmanim.base.shapes import (
    MatrixBox, VectorBar, LabeledBlock, MathLabel,
    ATTENTION_COLOR, QUERY_COLOR, KEY_COLOR, VALUE_COLOR,
    HIGHLIGHT_COLOR, DIM_COLOR, BACKGROUND_COLOR,
    TOKEN_COLOR, EMBEDDING_COLOR, FFN_COLOR,
)
from llmanim.base.animations import (
    data_flow_arrow, pulse_glow, layer_stack_anim,
    label_appear, highlight_sequence,
)
from llmanim.base.utils import (
    apply_dark_theme, make_attention_weights,
    attention_palette,
)
from llmanim.tokenization.token_box import _token_color


HEAD_COLORS = [
    "#E91E63", "#3F51B5", "#009688",
    "#FF9800", "#9C27B0", "#4CAF50",
    "#00BCD4", "#F44336",
]


class MultiHeadAttention(Scene):
    def construct(self):
        apply_dark_theme(self)
        N_HEADS = 4
        SEQ_LEN = 5

        # ══════════════════════════════════════════
        # SCENE 1 — Title
        # ══════════════════════════════════════════
        title = Text("Multi-Head Attention", font_size=44, color=WHITE, weight=BOLD)
        sub   = Text("One model, many perspectives", font_size=22, color=DIM_COLOR
                     ).next_to(title, DOWN, buff=0.3)
        self.play(Write(title))
        self.play(FadeIn(sub))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(sub))

        # ══════════════════════════════════════════
        # SCENE 2 — Why one head isn't enough
        # ══════════════════════════════════════════
        why_lbl = Text("Why one attention head isn't enough:", font_size=24,
                       color=WHITE, weight=BOLD).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(why_lbl))

        reasons = [
            ("Syntax",    "Subject → Verb agreement",          QUERY_COLOR),
            ("Reference", "Pronoun → antecedent (IT → animal)",KEY_COLOR),
            ("Position",  "Local context (nearby words)",       VALUE_COLOR),
            ("Semantics", "Long-range meaning relationships",   TOKEN_COLOR),
        ]
        reason_mobs = []
        for i, (name, desc, color) in enumerate(reasons):
            nm = Text(f"Head {i+1}: {name}", font_size=20, color=color, weight=BOLD)
            dm = Text(desc, font_size=16, color=WHITE)
            row = VGroup(nm, dm).arrange(RIGHT, buff=0.5)
            reason_mobs.append(row)

        reason_group = VGroup(*reason_mobs).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        reason_group.shift(DOWN * 0.2)
        for mob in reason_mobs:
            self.play(FadeIn(mob, shift=RIGHT * 0.1), run_time=0.4)
            self.wait(0.3)
        self.wait(1.0)
        self.play(FadeOut(reason_group), FadeOut(why_lbl))

        # ══════════════════════════════════════════
        # SCENE 3 — Splitting d_model into heads
        # ══════════════════════════════════════════
        split_lbl = Text("Step 1: Split d_model into h heads", font_size=22,
                         color=WHITE, weight=BOLD).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(split_lbl))

        # full d_model vector
        full_vec = VectorBar(dim=16, label="d_model = 512", color=EMBEDDING_COLOR)
        full_vec.shift(UP * 1.5)
        self.play(FadeIn(full_vec))
        self.wait(0.4)

        # split into 4 head vectors
        head_vecs = []
        for i in range(N_HEADS):
            hv = VectorBar(
                dim=4,
                label=f"Head {i+1}\nd_k=128",
                color=HEAD_COLORS[i],
            )
            head_vecs.append(hv)

        head_row = VGroup(*head_vecs).arrange(RIGHT, buff=0.45).shift(DOWN * 0.3)

        split_arrows = VGroup(*[
            Arrow(
                full_vec.get_bottom(),
                hv.get_top(),
                color=HEAD_COLORS[i],
                stroke_width=1.5,
                buff=0.1,
            )
            for i, hv in enumerate(head_vecs)
        ])
        self.play(
            AnimationGroup(*[GrowArrow(a) for a in split_arrows], lag_ratio=0.1)
        )
        self.play(
            AnimationGroup(*[FadeIn(hv) for hv in head_vecs], lag_ratio=0.1)
        )

        math_note = MathTex(
            r"d_k = d_{\text{model}} / h = 512 / 4 = 128",
            font_size=22, color=DIM_COLOR,
        ).to_edge(DOWN).shift(UP * 0.3)
        self.play(Write(math_note))
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ══════════════════════════════════════════
        # SCENE 4 — Each head runs independently
        # ══════════════════════════════════════════
        indep_lbl = Text("Step 2: Each head runs its own attention", font_size=22,
                         color=WHITE, weight=BOLD).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(indep_lbl))

        # 4 small attention blocks side by side
        head_blocks = []
        head_labels = []
        for i in range(N_HEADS):
            blk = LabeledBlock(
                f"Head {i+1}",
                width=1.7, height=0.75,
                color=HEAD_COLORS[i],
                sublabel="Q·Kᵀ/√d_k → V",
            )
            head_blocks.append(blk)
            head_labels.append(Text(f"H{i+1}", font_size=14, color=HEAD_COLORS[i]))

        blocks_row = VGroup(*head_blocks).arrange(RIGHT, buff=0.4).shift(DOWN * 0.1)
        self.play(AnimationGroup(*[FadeIn(b) for b in head_blocks], lag_ratio=0.12))

        # show different heatmaps per head
        attn_sets = [
            make_attention_weights(SEQ_LEN, 1, causal=False, seed=i)[0]
            for i in range(N_HEADS)
        ]
        heatmaps = []
        for i, w in enumerate(attn_sets):
            hm = MatrixBox(rows=SEQ_LEN, cols=SEQ_LEN, cell_size=0.22, color=HEAD_COLORS[i])
            for r in range(SEQ_LEN):
                for c in range(SEQ_LEN):
                    hm._cell(r, c).set_fill(
                        HEAD_COLORS[i],
                        opacity=float(np.clip(w[r, c] * 3.5, 0.05, 0.95)),
                    )
            hm.next_to(head_blocks[i], DOWN, buff=0.3)
            heatmaps.append(hm)

        self.play(AnimationGroup(*[h.fill_anim(lag=0.01) for h in heatmaps], lag_ratio=0.15))

        patterns_note = Text(
            "Each head learns a different attention pattern",
            font_size=18, color=DIM_COLOR,
        ).to_edge(DOWN).shift(UP * 0.2)
        self.play(FadeIn(patterns_note))
        self.wait(1.2)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ══════════════════════════════════════════
        # SCENE 5 — Concatenate all head outputs
        # ══════════════════════════════════════════
        concat_lbl = Text("Step 3: Concatenate all head outputs", font_size=22,
                          color=WHITE, weight=BOLD).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(concat_lbl))

        # 4 small output vectors
        out_vecs = [
            VectorBar(dim=4, label=f"O{i+1}", color=HEAD_COLORS[i])
            for i in range(N_HEADS)
        ]
        out_row = VGroup(*out_vecs).arrange(RIGHT, buff=0.3).shift(UP * 0.8)
        self.play(AnimationGroup(*[FadeIn(v) for v in out_vecs], lag_ratio=0.1))
        self.wait(0.4)

        # concatenate → long vector
        concat_vec = VectorBar(dim=16, label="Concat", color=HIGHLIGHT_COLOR)
        concat_vec.shift(DOWN * 0.5)
        concat_arrows = VGroup(*[
            Arrow(v.get_bottom(), concat_vec.get_top(), color=HEAD_COLORS[i],
                  stroke_width=1.5, buff=0.1)
            for i, v in enumerate(out_vecs)
        ])
        self.play(
            AnimationGroup(*[GrowArrow(a) for a in concat_arrows], lag_ratio=0.08)
        )
        self.play(FadeIn(concat_vec))

        shape_note = Text(
            "Shape: (seq_len, h × d_k) = (seq_len, d_model)",
            font_size=16, color=DIM_COLOR,
        ).to_edge(DOWN).shift(UP * 0.3)
        self.play(FadeIn(shape_note))
        self.wait(0.8)

        # ── Project with W_O ─────────────────────
        proj_lbl = Text("Step 4: Project with W_O → back to d_model", font_size=18,
                        color=DIM_COLOR).next_to(concat_vec, DOWN, buff=0.3)
        self.play(FadeIn(proj_lbl))

        W_O = MatrixBox(rows=4, cols=16, cell_size=0.22, color=EMBEDDING_COLOR)
        W_O.shift(DOWN * 2.2)
        W_O_lbl = Text("W_O", font_size=14, color=EMBEDDING_COLOR, weight=BOLD
                       ).next_to(W_O, UP, buff=0.1)
        self.play(W_O.fill_anim(lag=0.01), FadeIn(W_O_lbl))

        final_vec = VectorBar(dim=4, label="Output", color=EMBEDDING_COLOR)
        final_vec.shift(DOWN * 3.3)
        fa, faa = data_flow_arrow(W_O, final_vec, color=EMBEDDING_COLOR)
        self.add(fa); self.play(faa)
        self.play(FadeIn(final_vec))
        self.wait(1.2)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ══════════════════════════════════════════
        # SCENE 6 — Full MHA block diagram
        # ══════════════════════════════════════════
        full_lbl = Text("Multi-Head Attention — full picture", font_size=24,
                        color=WHITE, weight=BOLD).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(full_lbl))

        # input
        input_blk = LabeledBlock("Input X", width=2.2, height=0.6, color=EMBEDDING_COLOR)
        input_blk.shift(UP * 2.5)
        # linear projections
        wq_blk = LabeledBlock("W_Q", width=1.4, height=0.55, color=QUERY_COLOR)
        wk_blk = LabeledBlock("W_K", width=1.4, height=0.55, color=KEY_COLOR)
        wv_blk = LabeledBlock("W_V", width=1.4, height=0.55, color=VALUE_COLOR)
        proj_row = VGroup(wq_blk, wk_blk, wv_blk).arrange(RIGHT, buff=0.5).shift(UP * 1.3)
        # attention
        attn_blk = LabeledBlock("Scaled Dot-Product\nAttention ×h",
                                width=3.0, height=0.75, color=ATTENTION_COLOR)
        attn_blk.shift(DOWN * 0.2)
        # concat + project
        concat_blk = LabeledBlock("Concat + W_O", width=2.5, height=0.6, color=EMBEDDING_COLOR)
        concat_blk.shift(DOWN * 1.4)

        all_blks = [input_blk, wq_blk, wk_blk, wv_blk, attn_blk, concat_blk]
        self.play(AnimationGroup(*[FadeIn(b) for b in all_blks], lag_ratio=0.12))

        # connecting arrows
        for wb in [wq_blk, wk_blk, wv_blk]:
            a, an = data_flow_arrow(input_blk, wb, color=EMBEDDING_COLOR)
            self.add(a); self.play(an, run_time=0.3)

        for wb in [wq_blk, wk_blk, wv_blk]:
            a, an = data_flow_arrow(wb, attn_blk, color=wb.base_color)
            self.add(a); self.play(an, run_time=0.25)

        a, an = data_flow_arrow(attn_blk, concat_blk, color=ATTENTION_COLOR)
        self.add(a); self.play(an)
        self.wait(1.2)

        # ══════════════════════════════════════════
        # SCENE 7 — Formula + Summary
        # ══════════════════════════════════════════
        self.play(*[FadeOut(m) for m in self.mobjects])
        formula = MathTex(
            r"\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O",
            font_size=28, color=WHITE,
        ).shift(UP * 1.5)
        self.play(Write(formula))
        self.wait(0.8)

        summary = VGroup(*[
            Text(t, font_size=20, color=WHITE)
            for t in [
                "• Split d_model into h parallel heads",
                "• Each head = independent attention",
                "• Different heads learn different patterns",
                "• Concat outputs → project with W_O",
                "• Total compute ≈ single head  (d_k = d_model/h)",
            ]
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.28).shift(DOWN * 0.5)
        for item in summary:
            self.play(FadeIn(item, shift=RIGHT * 0.1), run_time=0.35)
            self.wait(0.2)
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])
        end = Text("Next → Layer Norm & Residuals", font_size=28, color=ATTENTION_COLOR)
        self.play(Write(end))
        self.wait(2)