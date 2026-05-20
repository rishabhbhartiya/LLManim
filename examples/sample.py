import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "manim_transformer"))
from manim import *
from base.shapes import MatrixBox, VectorBar, TokenBox
from base.animations import highlight_sequence, data_flow_arrow
from base.utils import apply_dark_theme, make_attention_weights
from tokenization.token_box import TokenRow
from base.shapes import TOKEN_COLOR, HIGHLIGHT_COLOR

class MyTransformerScene(Scene):
    def construct(self):
        apply_dark_theme(self)

        # 1. Show a token sequence
        tokens = ["the", "cat", "sat"]
        ids = [1, 2, 3]
        row = TokenRow(tokens, token_ids=ids)
        self.play(row.appear())

        # 2. Show an embedding matrix
        mat = MatrixBox(rows=4, cols=6, color=TOKEN_COLOR).shift(DOWN * 2)
        self.play(mat.fill_anim())

        # 3. Connect them with an animated arrow
        arrow, anim = data_flow_arrow(row, mat, label="embed")
        self.add(arrow)
        self.play(anim)

        # 4. Highlight components
        self.play(highlight_sequence(row.boxes))
        self.play(mat.highlight_row(2))
        self.wait(1)