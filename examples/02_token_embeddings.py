"""
VIDEO 2 — "Token Embeddings"
==============================
Runtime: ~4 minutes
Covers:
  - Why integers aren't enough (no meaning in numbers)
  - Embedding matrix lookup  (vocab_size × d_model)
  - Token ID → dense vector
  - What the numbers mean  (dimensions as features)
  - Similar tokens → similar vectors
  - Embedding addition (king - man + woman)

Run:
    manim -pqh 02_token_embeddings.py TokenEmbeddings
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import numpy as np
from base.shapes import (
    MatrixBox, VectorBar, TokenBox, LabeledBlock, MathLabel,
    TOKEN_COLOR, EMBEDDING_COLOR, HIGHLIGHT_COLOR,
    QUERY_COLOR, DIM_COLOR, BACKGROUND_COLOR,
)
from base.animations import (
    highlight_sequence, data_flow_arrow,
    vector_addition_anim, equation_reveal, label_appear,
)
from base.utils import (
    apply_dark_theme, make_embedding_matrix,
    make_token_sequence, DEMO_VOCAB, fmt_num,
)
from tokenization.token_box import (
    TokenRow, TokenIDMapping, _token_color,
)


class TokenEmbeddings(Scene):
    def construct(self):
        apply_dark_theme(self)

        # ══════════════════════════════════════════
        # SCENE 1 — Title
        # ══════════════════════════════════════════
        title = Text("Token Embeddings", font_size=48, color=WHITE, weight=BOLD)
        sub   = Text("Turning integers into meaning", font_size=22, color=DIM_COLOR
                     ).next_to(title, DOWN, buff=0.3)
        self.play(Write(title))
        self.play(FadeIn(sub, shift=UP * 0.1))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(sub))

        # ══════════════════════════════════════════
        # SCENE 2 — Problem: integers have no meaning
        # ══════════════════════════════════════════
        prob = Text(
            "Token IDs are just integers.\nDo 0 and 1 mean similar things?",
            font_size=28, color=WHITE, line_spacing=1.4,
        )
        self.play(Write(prob), run_time=1.0)
        self.wait(1.2)

        # show two IDs side by side
        id_a = Text("0", font_size=48, color=TOKEN_COLOR)
        id_b = Text("1", font_size=48, color=QUERY_COLOR)
        pair = VGroup(id_a, id_b).arrange(RIGHT, buff=1.2).shift(DOWN * 0.5)
        self.play(prob.animate.shift(UP * 1.5).scale(0.75))
        self.play(FadeIn(id_a), FadeIn(id_b))
        self.wait(0.5)

        cross = Text("≠ close in meaning!", font_size=22, color=HIGHLIGHT_COLOR
                     ).next_to(pair, DOWN, buff=0.3)
        self.play(FadeIn(cross))
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in [prob, pair, cross]])

        # ══════════════════════════════════════════
        # SCENE 3 — Solution: embedding matrix
        # ══════════════════════════════════════════
        sol = Text("Solution: embedding matrix", font_size=30, color=EMBEDDING_COLOR,
                   weight=BOLD).to_edge(UP).shift(DOWN * 0.2)
        self.play(Write(sol))

        # Show matrix with dim labels
        emb_mat = MatrixBox(
            rows=8, cols=10,
            cell_size=0.42,
            color=EMBEDDING_COLOR,
        ).shift(LEFT * 2.5 + DOWN * 0.3)
        self.play(emb_mat.fill_anim())

        # dimension labels
        row_brace = Brace(emb_mat, LEFT, buff=0.1)
        col_brace = Brace(emb_mat, DOWN, buff=0.1)
        row_lbl   = Text("vocab_size\n(50,257)", font_size=13, color=EMBEDDING_COLOR
                         ).next_to(row_brace, LEFT, buff=0.1)
        col_lbl   = Text("d_model (512)", font_size=13, color=EMBEDDING_COLOR
                         ).next_to(col_brace, DOWN, buff=0.1)
        self.play(
            Create(row_brace), Create(col_brace),
            FadeIn(row_lbl), FadeIn(col_lbl),
        )
        self.wait(0.6)

        # label the matrix
        mat_label = Text("Embedding Matrix  E", font_size=18, color=EMBEDDING_COLOR
                         ).next_to(emb_mat, UP, buff=0.2)
        self.play(FadeIn(mat_label))
        self.wait(0.8)

        # ══════════════════════════════════════════
        # SCENE 4 — Row lookup animation
        # ══════════════════════════════════════════
        # token "cat" = ID 2  → highlight row 2
        tok_box = TokenBox("cat", token_id=2, color=TOKEN_COLOR)
        tok_box.next_to(emb_mat, RIGHT, buff=1.8).shift(UP * 1.0)
        self.play(FadeIn(tok_box))

        arrow1, anim1 = data_flow_arrow(
            tok_box, emb_mat,
            label="ID = 2",
            color=TOKEN_COLOR,
        )
        self.add(arrow1)
        self.play(anim1)
        self.play(emb_mat.highlight_row(2, color=HIGHLIGHT_COLOR))
        self.wait(0.4)

        # extract row → vector
        vec = VectorBar(
            dim=10,
            values=np.random.uniform(-1, 1, 10),
            label="e_cat",
            color=EMBEDDING_COLOR,
        )
        target_pos = emb_mat.get_center() + RIGHT * (emb_mat.width / 2 + 1.5) + DOWN * 0.3
        vec.shift(target_pos)

        self.play(FadeIn(vec))
        arrow2, anim2 = data_flow_arrow(
            emb_mat, vec,
            label="row 2",
            color=EMBEDDING_COLOR,
        )
        self.add(arrow2)
        self.play(anim2)
        self.wait(0.5)

        note = Text(
            "Each row = one token's embedding vector",
            font_size=17, color=DIM_COLOR,
        ).to_edge(DOWN).shift(UP * 0.2)
        self.play(FadeIn(note))
        self.wait(1.2)
        self.play(*[FadeOut(m) for m in [
            emb_mat, row_brace, col_brace, row_lbl, col_lbl,
            mat_label, tok_box, arrow1, arrow2, vec, note, sol,
        ]])

        # ══════════════════════════════════════════
        # SCENE 5 — What do the dimensions mean?
        # ══════════════════════════════════════════
        dim_lbl = Text("What do the dimensions represent?", font_size=28,
                       color=WHITE, weight=BOLD).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(dim_lbl))

        # Three vectors side by side for cat, dog, king
        words  = ["cat", "dog", "king"]
        vecs   = [
            np.array([0.9, 0.8, -0.1, 0.2, 0.3, -0.7, 0.1, 0.4]),
            np.array([0.8, 0.85, -0.05, 0.15, 0.25, -0.6, 0.15, 0.35]),
            np.array([-0.1, -0.2, 0.9, 0.85, -0.3, 0.1, 0.8, 0.7]),
        ]
        colors = [TOKEN_COLOR, EMBEDDING_COLOR, QUERY_COLOR]
        vec_mobs = []
        for i, (w, v, c) in enumerate(zip(words, vecs, colors)):
            vb = VectorBar(dim=8, values=v, label=w, color=c)
            vec_mobs.append(vb)

        row = VGroup(*vec_mobs).arrange(RIGHT, buff=0.7).shift(DOWN * 0.3)
        self.play(AnimationGroup(*[FadeIn(v) for v in vec_mobs], lag_ratio=0.2))
        self.wait(0.5)

        # highlight "cat" and "dog" as similar
        self.play(
            vec_mobs[0].pulse(),
            vec_mobs[1].pulse(),
        )
        sim_note = Text("cat & dog: similar vectors  (both animals)", font_size=18,
                        color=HIGHLIGHT_COLOR).to_edge(DOWN).shift(UP * 0.2)
        self.play(FadeIn(sim_note))
        self.wait(1.0)
        self.play(FadeOut(sim_note))

        # highlight "king" as different
        self.play(vec_mobs[2].pulse(color=QUERY_COLOR))
        diff_note = Text("king: different direction  (royalty, not animal)", font_size=18,
                         color=QUERY_COLOR).to_edge(DOWN).shift(UP * 0.2)
        self.play(FadeIn(diff_note))
        self.wait(1.2)
        self.play(*[FadeOut(m) for m in vec_mobs + [dim_lbl, diff_note]])

        # ══════════════════════════════════════════
        # SCENE 6 — Famous analogy: king - man + woman
        # ══════════════════════════════════════════
        analogy_lbl = Text("The famous analogy:", font_size=26, color=WHITE, weight=BOLD
                           ).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(analogy_lbl))

        eq_terms = [
            r"\text{king}", "-", r"\text{man}", "+", r"\text{woman}",
            "\\approx", r"\text{queen}",
        ]
        eq_colors = [QUERY_COLOR, WHITE, DIM_COLOR, WHITE, EMBEDDING_COLOR, WHITE, TOKEN_COLOR]
        eq_mobs = VGroup(*[
            MathTex(t, font_size=36, color=c)
            for t, c in zip(eq_terms, eq_colors)
        ]).arrange(RIGHT, buff=0.2).shift(DOWN * 0.2)

        for mob in eq_mobs:
            self.play(FadeIn(mob, shift=UP * 0.1), run_time=0.3)
            self.wait(0.3)
        self.wait(0.8)

        sub_note = Text(
            "Vector arithmetic in embedding space captures real-world meaning",
            font_size=17, color=DIM_COLOR,
        ).to_edge(DOWN).shift(UP * 0.2)
        self.play(FadeIn(sub_note))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ══════════════════════════════════════════
        # SCENE 7 — Full token sequence → embedding matrix → vectors
        # ══════════════════════════════════════════
        full_lbl = Text("Full sequence embedding", font_size=26, color=WHITE, weight=BOLD
                        ).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(full_lbl))

        tokens, ids = make_token_sequence(["the", "cat", "sat"])
        tok_row = TokenRow(tokens, token_ids=ids, show_ids=True)
        tok_row.shift(UP * 1.5)
        self.play(tok_row.appear())
        self.wait(0.3)

        # small embedding matrix in center
        small_mat = MatrixBox(rows=5, cols=6, cell_size=0.38, color=EMBEDDING_COLOR)
        small_mat.move_to(ORIGIN)
        mat_l = Text("E", font_size=18, color=EMBEDDING_COLOR).next_to(small_mat, UP, buff=0.15)
        self.play(small_mat.fill_anim(), FadeIn(mat_l))

        arr_in, anim_in = data_flow_arrow(tok_row, small_mat, color=TOKEN_COLOR)
        self.add(arr_in)
        self.play(anim_in)

        # output vectors below
        out_vecs = VGroup(*[
            VectorBar(dim=6, label=t, color=_token_color(i))
            for i, t in enumerate(tokens)
        ]).arrange(RIGHT, buff=0.3).shift(DOWN * 2.0)
        self.play(FadeIn(out_vecs))

        arr_out, anim_out = data_flow_arrow(small_mat, out_vecs, color=EMBEDDING_COLOR)
        self.add(arr_out)
        self.play(anim_out)

        self.wait(1.0)

        shape_note = Text(
            "Output shape: (seq_len × d_model)  e.g.  (5 × 512)",
            font_size=17, color=DIM_COLOR,
        ).to_edge(DOWN).shift(UP * 0.2)
        self.play(FadeIn(shape_note))
        self.wait(1.5)

        # ══════════════════════════════════════════
        # SCENE 8 — Summary
        # ══════════════════════════════════════════
        self.play(*[FadeOut(m) for m in self.mobjects])
        summary = VGroup(*[
            Text(t, font_size=21, color=WHITE)
            for t in [
                "• Token ID  →  row lookup in embedding matrix",
                "• Each token gets a dense vector of floats",
                "• Similar tokens → similar vectors",
                "• Output shape: (seq_len, d_model)",
                "• The matrix is learned during training",
            ]
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.3).shift(RIGHT * 0.3)
        stitle = Text("Summary", font_size=28, color=EMBEDDING_COLOR, weight=BOLD
                      ).next_to(summary, UP, buff=0.4)
        self.play(FadeIn(stitle))
        for item in summary:
            self.play(FadeIn(item, shift=RIGHT * 0.1), run_time=0.35)
            self.wait(0.25)
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])
        end = Text("Next → Positional Encoding", font_size=28, color=QUERY_COLOR)
        self.play(Write(end))
        self.wait(2)