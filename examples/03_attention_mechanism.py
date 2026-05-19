"""
VIDEO 3 — "Attention Mechanism"
=================================
Runtime: ~5–6 minutes
Covers:
  - Why attention? (problem with fixed context)
  - Q K V intuition  (query / key / value analogy)
  - Linear projections  W_Q  W_K  W_V
  - Dot product scores
  - Scaling by √d_k
  - Softmax → attention weights
  - Weighted sum of Values
  - Attention heatmap

Run:
    manim -pqh 03_attention_mechanism.py AttentionMechanism
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import numpy as np
from base.shapes import (
    MatrixBox, VectorBar, TokenBox, LabeledBlock,
    MathLabel, SoftmaxCurve,
    TOKEN_COLOR, EMBEDDING_COLOR, ATTENTION_COLOR,
    QUERY_COLOR, KEY_COLOR, VALUE_COLOR,
    HIGHLIGHT_COLOR, DIM_COLOR, BACKGROUND_COLOR,
)
from base.animations import (
    data_flow_arrow, highlight_sequence, equation_reveal,
    pulse_glow, label_appear, attention_flow, zoom_into,
    softmax_temperature_sweep,
)
from base.utils import (
    apply_dark_theme, scaled_dot_product, causal_mask,
    make_attention_weights, make_weight_matrix,
    attention_palette, softmax,
)
from tokenization.token_box import TokenRow, _token_color


class AttentionMechanism(Scene):
    def construct(self):
        apply_dark_theme(self)

        # ══════════════════════════════════════════
        # SCENE 1 — Title
        # ══════════════════════════════════════════
        title = Text("The Attention Mechanism", font_size=44, color=WHITE, weight=BOLD)
        sub   = Text("How tokens talk to each other", font_size=22, color=DIM_COLOR
                     ).next_to(title, DOWN, buff=0.3)
        self.play(Write(title))
        self.play(FadeIn(sub))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(sub))

        # ══════════════════════════════════════════
        # SCENE 2 — The problem: why do we need attention?
        # ══════════════════════════════════════════
        example = Text(
            '"The animal didn\'t cross the street because IT was tired"',
            font_size=22, color=WHITE,
        ).shift(UP * 1.2)
        self.play(Write(example))
        self.wait(0.5)

        question = Text(
            "What does  IT  refer to?",
            font_size=26, color=HIGHLIGHT_COLOR,
        ).shift(DOWN * 0.2)
        self.play(FadeIn(question))
        self.wait(0.8)

        answer = Text(
            "The model must connect  IT  ←→  animal",
            font_size=22, color=TOKEN_COLOR,
        ).shift(DOWN * 1.0)
        self.play(FadeIn(answer))
        self.wait(1.2)
        self.play(*[FadeOut(m) for m in [example, question, answer]])

        # ══════════════════════════════════════════
        # SCENE 3 — Q K V intuition
        # ══════════════════════════════════════════
        intuition_lbl = Text("Query · Key · Value  Intuition", font_size=28,
                              color=WHITE, weight=BOLD).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(intuition_lbl))

        analogy_items = [
            ("Query  (Q)", "What am I looking for?",      QUERY_COLOR),
            ("Key    (K)", "What information do I have?",  KEY_COLOR),
            ("Value  (V)", "What do I actually return?",   VALUE_COLOR),
        ]
        analogy_mobs = []
        for i, (term, desc, color) in enumerate(analogy_items):
            term_mob = Text(term, font_size=22, color=color, weight=BOLD)
            desc_mob = Text(desc, font_size=18, color=WHITE)
            row = VGroup(term_mob, desc_mob).arrange(RIGHT, buff=0.4)
            analogy_mobs.append(row)

        analogy_group = VGroup(*analogy_mobs).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        analogy_group.shift(DOWN * 0.3)
        for mob in analogy_mobs:
            self.play(FadeIn(mob, shift=RIGHT * 0.15), run_time=0.45)
            self.wait(0.4)
        self.wait(1.0)
        self.play(FadeOut(analogy_group), FadeOut(intuition_lbl))

        # ══════════════════════════════════════════
        # SCENE 4 — Token sequence setup
        # ══════════════════════════════════════════
        seq_label = Text("Input token sequence:", font_size=18, color=DIM_COLOR
                         ).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(seq_label))

        words  = ["the", "animal", "didn't", "cross", "IT"]
        ids    = list(range(len(words)))
        tok_row = TokenRow(words, token_ids=ids, show_ids=False)
        tok_row.shift(UP * 2.2)
        self.play(tok_row.appear())
        self.wait(0.4)

        # ══════════════════════════════════════════
        # SCENE 5 — Linear projections W_Q W_K W_V
        # ══════════════════════════════════════════
        proj_label = Text("Step 1: Linear projections", font_size=20, color=DIM_COLOR
                          ).next_to(tok_row, DOWN, buff=0.5)
        self.play(FadeIn(proj_label))

        # input matrix (seq_len × d_model)
        X_mat = MatrixBox(rows=5, cols=6, cell_size=0.34, color=EMBEDDING_COLOR)
        X_mat.shift(LEFT * 4.5 + DOWN * 0.3)
        X_lbl = Text("X", font_size=16, color=EMBEDDING_COLOR).next_to(X_mat, UP, buff=0.1)
        self.play(X_mat.fill_anim(lag=0.015), FadeIn(X_lbl))

        # three weight matrices
        w_data = [
            ("W_Q", QUERY_COLOR, LEFT * 1.5),
            ("W_K", KEY_COLOR,   ORIGIN),
            ("W_V", VALUE_COLOR, RIGHT * 1.5),
        ]
        w_mats, w_lbls = [], []
        for name, color, shift in w_data:
            wm = MatrixBox(rows=6, cols=4, cell_size=0.28, color=color)
            wm.shift(shift + DOWN * 0.3)
            wl = Text(name, font_size=14, color=color, weight=BOLD).next_to(wm, UP, buff=0.1)
            w_mats.append(wm)
            w_lbls.append(wl)
            self.play(wm.fill_anim(lag=0.01), FadeIn(wl), run_time=0.4)

        # output Q K V
        qkv_data = [
            ("Q", QUERY_COLOR, LEFT * 1.5 + DOWN * 2.5),
            ("K", KEY_COLOR,   ORIGIN     + DOWN * 2.5),
            ("V", VALUE_COLOR, RIGHT * 1.5+ DOWN * 2.5),
        ]
        qkv_mats, qkv_lbls = [], []
        for name, color, shift in qkv_data:
            qm = MatrixBox(rows=5, cols=4, cell_size=0.28, color=color)
            qm.shift(shift)
            ql = Text(name, font_size=14, color=color, weight=BOLD).next_to(qm, UP, buff=0.1)
            qkv_mats.append(qm)
            qkv_lbls.append(ql)

        # arrows X → W_Q/W_K/W_V → Q/K/V
        for i, (wm, qm, (name, color, _)) in enumerate(zip(w_mats, qkv_mats, qkv_data)):
            a1, an1 = data_flow_arrow(X_mat, wm, color=color)
            self.add(a1); self.play(an1, run_time=0.4)
            self.play(wm.highlight_row(2, color=HIGHLIGHT_COLOR), run_time=0.2)
            a2, an2 = data_flow_arrow(wm, qm, color=color)
            self.add(a2); self.play(an2, run_time=0.35)
            self.play(qm.fill_anim(lag=0.01), FadeIn(qkv_lbls[i]), run_time=0.35)

        self.wait(0.8)
        fade_list = (w_mats + w_lbls + qkv_lbls + [X_mat, X_lbl, proj_label, seq_label, tok_row])
        self.play(*[FadeOut(m) for m in fade_list])

        # ══════════════════════════════════════════
        # SCENE 6 — Dot product attention scores
        # ══════════════════════════════════════════
        score_label = Text("Step 2: Attention Scores  Q · Kᵀ", font_size=22,
                           color=WHITE, weight=BOLD).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(score_label))

        Q_mat = qkv_mats[0].copy().shift(LEFT * 3.5 + UP * 0.5)
        K_mat = qkv_mats[1].copy().shift(LEFT * 0.5 + UP * 0.5)
        Q_l   = Text("Q", font_size=18, color=QUERY_COLOR, weight=BOLD).next_to(Q_mat, UP, buff=0.1)
        K_l   = Text("Kᵀ", font_size=18, color=KEY_COLOR, weight=BOLD).next_to(K_mat, UP, buff=0.1)

        self.play(FadeIn(Q_mat), FadeIn(K_mat), FadeIn(Q_l), FadeIn(K_l))

        times_t = Text("×", font_size=28, color=WHITE
                       ).move_to(midpoint(Q_mat.get_right(), K_mat.get_left()))
        self.play(FadeIn(times_t))

        # score matrix
        score_mat = MatrixBox(rows=5, cols=5, cell_size=0.36, color=ATTENTION_COLOR)
        score_mat.shift(RIGHT * 3.0 + UP * 0.5)
        eq_t = Text("=", font_size=28, color=WHITE
                    ).move_to(midpoint(K_mat.get_right(), score_mat.get_left()))
        self.play(FadeIn(eq_t))
        self.play(score_mat.fill_anim(lag=0.02))
        S_l = Text("Scores", font_size=14, color=ATTENTION_COLOR).next_to(score_mat, UP, buff=0.1)
        self.play(FadeIn(S_l))
        self.wait(0.6)

        # ── Scale by √d_k ────────────────────────
        scale_note = Text("Step 3: Divide by √d_k  (prevents vanishing gradients)",
                          font_size=17, color=DIM_COLOR).shift(DOWN * 1.8)
        sqrt_eq = MathTex(r"\frac{QK^T}{\sqrt{d_k}}", font_size=32, color=ATTENTION_COLOR
                          ).shift(DOWN * 2.6)
        self.play(FadeIn(scale_note))
        self.play(Write(sqrt_eq))
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in [
            Q_mat, K_mat, Q_l, K_l, times_t, eq_t, S_l, score_label,
            scale_note, sqrt_eq,
        ]])

        # ══════════════════════════════════════════
        # SCENE 7 — Softmax → attention weights
        # ══════════════════════════════════════════
        soft_label = Text("Step 4: Softmax → attention weights", font_size=22,
                          color=WHITE, weight=BOLD).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(soft_label))

        logits = np.array([1.2, 3.5, 0.4, -0.8, 0.9])
        sc = SoftmaxCurve(
            logits=logits,
            labels=["the", "animal", "didn't", "cross", "IT"],
            temperature=1.0,
            color=ATTENTION_COLOR,
        ).shift(LEFT * 1.5)
        sc_lbl = Text("Query token: IT", font_size=16, color=HIGHLIGHT_COLOR
                      ).next_to(sc, UP, buff=0.2)
        self.play(FadeIn(sc), FadeIn(sc_lbl))
        self.play(sc.show_values())
        self.wait(0.5)

        # animate temperature change
        softmax_temperature_sweep(sc, t_start=1.0, t_end=0.3, steps=6, scene=self)
        self.wait(0.5)
        softmax_temperature_sweep(sc, t_start=0.3, t_end=2.5, steps=6, scene=self)
        self.wait(0.8)

        attn_note = Text(
            "High weight = this token matters more for the query",
            font_size=17, color=DIM_COLOR,
        ).to_edge(DOWN).shift(UP * 0.2)
        self.play(FadeIn(attn_note))
        self.wait(1.2)
        self.play(*[FadeOut(m) for m in [sc, sc_lbl, soft_label, attn_note]])

        # ══════════════════════════════════════════
        # SCENE 8 — Attention heatmap
        # ══════════════════════════════════════════
        heat_label = Text("Attention heatmap  (all tokens × all tokens)", font_size=20,
                          color=WHITE, weight=BOLD).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(heat_label))

        # 5×5 heatmap using MatrixBox with color-coded values
        attn_weights = make_attention_weights(seq_len=5, n_heads=1, causal=False, seed=7)[0]
        heat_mat = MatrixBox(rows=5, cols=5, cell_size=0.55, color=ATTENTION_COLOR)
        heat_mat.shift(DOWN * 0.3)

        # color each cell by attention weight
        for r in range(5):
            for c in range(5):
                w = float(attn_weights[r, c])
                heat_mat._cell(r, c).set_fill(
                    color=ATTENTION_COLOR,
                    opacity=float(np.clip(w * 3, 0.05, 0.95)),
                )
        self.play(heat_mat.fill_anim(lag=0.03))

        # axis labels
        tok_labels = ["the", "animal", "didn't", "cross", "IT"]
        for i, lbl in enumerate(tok_labels):
            t = Text(lbl, font_size=11, color=DIM_COLOR)
            t.next_to(heat_mat._cell(i, 0), LEFT, buff=0.1)
            t2 = Text(lbl, font_size=11, color=DIM_COLOR)
            t2.next_to(heat_mat._cell(0, i), UP, buff=0.1)
            self.add(t, t2)

        self.wait(0.5)

        # highlight IT row
        self.play(heat_mat.highlight_row(4, color=HIGHLIGHT_COLOR))
        it_note = Text(
            '"IT" attends strongly to "animal"',
            font_size=18, color=HIGHLIGHT_COLOR,
        ).to_edge(DOWN).shift(UP * 0.2)
        self.play(FadeIn(it_note))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ══════════════════════════════════════════
        # SCENE 9 — Weighted sum of Values
        # ══════════════════════════════════════════
        ws_label = Text("Step 5: Weighted sum of Values", font_size=22,
                        color=WHITE, weight=BOLD).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(ws_label))

        # show weights row
        weights_row = VGroup(*[
            Rectangle(
                width=0.5, height=float(attn_weights[4, j]) * 2 + 0.1,
                fill_color=ATTENTION_COLOR,
                fill_opacity=float(np.clip(attn_weights[4, j] * 3, 0.1, 0.9)),
                stroke_width=1,
            )
            for j in range(5)
        ]).arrange(RIGHT, buff=0.1).shift(LEFT * 3.5 + UP * 0.5)
        w_lbl = Text("Attention weights (IT row)", font_size=14, color=ATTENTION_COLOR
                     ).next_to(weights_row, UP, buff=0.2)

        V_mat = MatrixBox(rows=5, cols=4, cell_size=0.38, color=VALUE_COLOR)
        V_mat.shift(RIGHT * 0.5 + UP * 0.5)
        V_lbl = Text("V", font_size=16, color=VALUE_COLOR, weight=BOLD
                     ).next_to(V_mat, UP, buff=0.1)

        out_vec = VectorBar(dim=4, label="output", color=HIGHLIGHT_COLOR)
        out_vec.shift(RIGHT * 3.8 + UP * 0.5)

        self.play(FadeIn(weights_row), FadeIn(w_lbl))
        self.play(V_mat.fill_anim(), FadeIn(V_lbl))

        a1, an1 = data_flow_arrow(weights_row, V_mat, color=ATTENTION_COLOR)
        self.add(a1); self.play(an1)

        a2, an2 = data_flow_arrow(V_mat, out_vec, label="weighted sum", color=VALUE_COLOR)
        self.add(a2); self.play(an2)
        self.play(FadeIn(out_vec))
        self.wait(0.5)

        ws_note = Text(
            "Each output position is a blend of all Value vectors,\nweighted by attention",
            font_size=17, color=DIM_COLOR, line_spacing=1.3,
        ).to_edge(DOWN).shift(UP * 0.3)
        self.play(FadeIn(ws_note))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ══════════════════════════════════════════
        # SCENE 10 — Full formula + summary
        # ══════════════════════════════════════════
        formula = MathTex(
            r"\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            font_size=34, color=WHITE,
        ).shift(UP * 1.0)
        self.play(Write(formula))
        self.play(pulse_glow(formula, color=ATTENTION_COLOR))
        self.wait(0.8)

        summary = VGroup(*[
            Text(t, font_size=20, color=WHITE)
            for t in [
                "• Q = what I'm looking for",
                "• K = what each token offers",
                "• V = what each token returns",
                "• Attention weight = similarity(Q, K)",
                "• Output = weighted sum of V",
            ]
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.28).shift(DOWN * 1.0)
        for item in summary:
            self.play(FadeIn(item, shift=RIGHT * 0.1), run_time=0.35)
            self.wait(0.2)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])
        end = Text("Next → Multi-Head Attention", font_size=28, color=ATTENTION_COLOR)
        self.play(Write(end))
        self.wait(2)


def midpoint(a, b):
    import numpy as np
    return (np.array(a) + np.array(b)) / 2