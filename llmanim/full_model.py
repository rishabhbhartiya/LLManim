"""
full_model.py
==============
Top-level composer assets for visualising a complete Transformer / LLM.

This file sits at the top of the llmanim asset hierarchy.
It imports from every sublayer module and assembles them into bird's-eye-view
visuals that show the whole model at once.

Assets
------
FullModelOverview      — The entire model as a single labelled-box diagram.
                         Zoomable into any component. Compact ↔ detailed toggle.
EndToEndAnim           — One complete forward pass: text → tokens → embeddings
                         → N layers → logits → sampled token → output text.
InferenceLoopAnim      — Wraps EndToEndAnim in an autoregressive loop.
                         Shows context window growing and KV cache filling.
ModelComparisonTable   — Animated table comparing GPT-2 / GPT-3 / LLaMA / BERT.
ArchitectureSelector   — Side-by-side Encoder-only / Decoder-only / Enc-Dec
                         with per-task annotations.

NOT in this file
----------------
  • Scene.construct() methods   — those live in your scene files
  • self.play() / self.wait()   — same
  • if __name__ == "__main__"   — same

Usage
-----
    # in your scene file, e.g. scenes/gpt_explained.py
    from manim import Scene
    from full_model import FullModelOverview, EndToEndAnim

    class GPTScene(Scene):
        def construct(self):
            overview = FullModelOverview(n_layers=12, style="gpt")
            self.play(overview.appear_animation())
            self.play(overview.zoom_into("Attention"))
            self.wait()
"""

from __future__ import annotations

from manim import (
    VGroup, Rectangle, RoundedRectangle, Text, MathTex, Tex,
    Arrow, Line, DashedLine, Brace,
    Scene, Animation, AnimationGroup, Succession,
    FadeIn, FadeOut, Create, Write, Uncreate,
    Transform, ReplacementTransform, TransformFromCopy,
    Indicate, Flash, Circumscribe,
    LaggedStart,
    GrowArrow, MoveToTarget,
    LEFT, RIGHT, UP, DOWN, ORIGIN,
    WHITE, GRAY, DARK_GRAY, LIGHT_GRAY, BLACK,
    interpolate_color,
    rate_functions, PI,
    Dot,
)

import numpy as _np

# ---------------------------------------------------------------------------
# Style imports — graceful fallback
# ---------------------------------------------------------------------------
try:
    from styles.colors import COLORS
except ImportError:
    COLORS = {
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

try:
    from styles.constants import LABEL_SCALE, DEFAULT_RUN_TIME, FAST_RUN_TIME
except ImportError:
    LABEL_SCALE      = 0.35
    DEFAULT_RUN_TIME = 1.0
    FAST_RUN_TIME    = 0.4

# ---------------------------------------------------------------------------
# Downstream asset imports — each is guarded so the file parses even when
# a sibling module hasn't been written yet.
# ---------------------------------------------------------------------------
try:
    from blocks.transformer_block import (
        TransformerBlockDiagram, LayerStackAnim,
        GPTBlockAnim, EncoderBlockAnim, DecoderBlockAnim,
    )
    _HAS_BLOCKS = True
except ImportError:
    _HAS_BLOCKS = False

try:
    from embeddings.embedding_lookup import (
        EmbeddingMatrix, TokenToVectorAnim, VectorDisplay,
    )
    _HAS_EMBED = True
except ImportError:
    _HAS_EMBED = False

try:
    from embeddings.positional_encoding import (
        PEAdditionAnim, PositionBadges,
    )
    _HAS_PE = True
except ImportError:
    _HAS_PE = False

try:
    from feedforward.ffn import FFNBlock, ActivationCurve
    _HAS_FFN = True
except ImportError:
    _HAS_FFN = False


# ---------------------------------------------------------------------------
# Shared internal helpers
# ---------------------------------------------------------------------------

def _txt(s: str, color=LIGHT_GRAY, scale: float = LABEL_SCALE) -> Text:
    return Text(s, font="Monospace").scale(scale).set_color(color)


def _component_box(
    label:        str,
    color:        str,
    width:        float = 3.2,
    height:       float = 0.60,
    fill_opacity: float = 0.80,
    sublabel:     str   = "",
) -> VGroup:
    """
    A rounded rectangle with a primary label and an optional smaller sublabel.
    Used as the building block for FullModelOverview.
    """
    rect = RoundedRectangle(
        width=width, height=height,
        corner_radius=0.08,
        fill_color=color, fill_opacity=fill_opacity,
        stroke_color=WHITE, stroke_width=1.1,
    )
    main_lbl = Text(label, font="Monospace") \
                   .scale(LABEL_SCALE).set_color(WHITE) \
                   .move_to(rect.get_center())
    group = VGroup(rect, main_lbl)

    if sublabel:
        sub = Text(sublabel, font="Monospace") \
                  .scale(LABEL_SCALE - 0.07).set_color(LIGHT_GRAY) \
                  .next_to(rect, RIGHT, buff=0.18)
        group.add(sub)

    group._rect  = rect
    group._label = main_lbl
    group._color = color
    return group


def _flow_arrow(start_mob, end_mob, color=COLORS["arrow"]) -> Arrow:
    return Arrow(
        start_mob.get_bottom(), end_mob.get_top(),
        buff=0.06, color=color,
        stroke_width=1.4,
        max_tip_length_to_length_ratio=0.28,
    )


# ===========================================================================
# 1.  FullModelOverview
# ===========================================================================

class FullModelOverview(VGroup):
    """
    The entire Transformer / LLM rendered as a single vertical stack of
    labelled component boxes with connecting arrows.

    Supports three style presets:
        "gpt"     — GPT-style decoder-only (token emb → N× block → LM head)
        "bert"    — Encoder-only           (token emb + segment → N× block → pooler)
        "enc_dec" — Encoder-Decoder        (encoder stack + decoder stack)

    Parameters
    ----------
    n_layers   : int   — number of Transformer blocks shown in the stack
    style      : str   — 'gpt' | 'bert' | 'enc_dec'
    box_width  : float — width of each component box
    show_params: bool  — annotate each box with a rough parameter count
    """

    # Component definitions per style:
    # (label, color, sublabel_template)
    _COMPONENTS = {
        "gpt": [
            ("Input Text",         COLORS["token"],     ""),
            ("Tokenizer",          COLORS["token"],     "BPE / WordPiece"),
            ("Token Embedding",    COLORS["embedding"], "vocab × d_model"),
            ("Positional Encoding",COLORS["norm"],      "sinusoidal / RoPE"),
            ("{n}×  Transformer Block", COLORS["attention"], "LN → MHA → LN → FFN"),
            ("LM Head",            COLORS["output"],    "d_model → vocab"),
            ("Softmax + Sample",   COLORS["output"],    "next token"),
            ("Output Text",        COLORS["token"],     ""),
        ],
        "bert": [
            ("Input Text",         COLORS["token"],     ""),
            ("Tokenizer",          COLORS["token"],     "WordPiece"),
            ("Token + Seg Embed",  COLORS["embedding"], "tok + seg + pos"),
            ("{n}×  Encoder Block",COLORS["attention"], "LN → MHA → LN → FFN"),
            ("Pooler",             COLORS["norm"],      "[CLS] → 768"),
            ("Task Head",          COLORS["output"],    "cls / ner / qa …"),
        ],
        "enc_dec": [
            ("Source Tokens",      COLORS["token"],     ""),
            ("Encoder Embedding",  COLORS["embedding"], "tok + pos"),
            ("{n}×  Encoder Block",COLORS["key"],       "MHA → FFN"),
            ("Encoder Output",     COLORS["value"],     "K, V passed →"),
            ("Target Tokens",      COLORS["token"],     "(shifted right)"),
            ("Decoder Embedding",  COLORS["embedding"], "tok + pos"),
            ("{n}×  Decoder Block",COLORS["attention"], "MaskedMHA → CrossAttn → FFN"),
            ("LM Head",            COLORS["output"],    "d_model → vocab"),
        ],
    }

    def __init__(
        self,
        n_layers:    int   = 12,
        style:       str   = "gpt",
        box_width:   float = 3.6,
        show_params: bool  = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.style    = style
        self.n_layers = n_layers

        spec = self._COMPONENTS.get(style, self._COMPONENTS["gpt"])

        self._boxes:  list[VGroup] = []
        self._arrows: list[Arrow]  = []
        self._labels: dict[str, VGroup] = {}  # label → box, for zoom_into

        BOX_GAP = 0.28

        # ── build boxes top → bottom ─────────────────────────────────────────
        for raw_label, color, sublabel in spec:
            label = raw_label.format(n=n_layers)
            box   = _component_box(label, color, width=box_width, sublabel=sublabel)
            self._boxes.append(box)
            self._labels[label] = box

        # vertical layout
        y = 0.0
        for box in self._boxes:
            box.move_to(_np.array([0.0, -y, 0]))
            y += box._rect.height + BOX_GAP

        # centre vertically
        total_h = y - BOX_GAP
        for box in self._boxes:
            box.shift(UP * total_h / 2)

        self.add(*self._boxes)

        # ── connecting arrows ─────────────────────────────────────────────────
        for i in range(len(self._boxes) - 1):
            arr = _flow_arrow(self._boxes[i], self._boxes[i + 1])
            self._arrows.append(arr)
            self.add(arr)

        # ── side annotations (parameter counts, rough) ────────────────────────
        if show_params:
            self._param_labels = self._build_param_annotations(style, n_layers, box_width)
            self.add(*self._param_labels)
        else:
            self._param_labels = []

        # ── title ─────────────────────────────────────────────────────────────
        style_names = {"gpt": "GPT-style  (Decoder-only)",
                       "bert": "BERT-style  (Encoder-only)",
                       "enc_dec": "Encoder-Decoder  (T5 / BART)"}
        title = _txt(style_names.get(style, style), COLORS["highlight"],
                     scale=LABEL_SCALE + 0.07)
        title.next_to(self, UP, buff=0.45)
        self.add(title)
        self._title = title

    # ── internal ─────────────────────────────────────────────────────────────

    def _build_param_annotations(
        self, style: str, n: int, box_w: float
    ) -> list[VGroup]:
        """
        Rough parameter count annotations placed to the right of relevant boxes.
        Numbers are illustrative (GPT-2 medium scale).
        """
        d  = 1024    # d_model
        V  = 50_000  # vocab size
        ff = 4 * d

        counts = {
            "gpt":  {
                "Token Embedding":     V * d,
                f"{n}×  Transformer Block": n * (4 * d * d + 2 * d * ff),
                "LM Head":             d * V,
            },
            "bert": {
                "Token + Seg Embed":   (V + 3) * d,
                f"{n}×  Encoder Block": n * (4 * d * d + 2 * d * ff),
            },
        }

        labels = []
        style_counts = counts.get(style, {})
        for box in self._boxes:
            raw = box._label.text
            params = style_counts.get(raw)
            if params:
                m = params / 1_000_000
                s = f"{m:.0f} M params" if m >= 1 else f"{params/1000:.0f} K params"
                lbl = _txt(s, COLORS["dim"], scale=LABEL_SCALE - 0.06)
                lbl.next_to(box, RIGHT, buff=0.22)
                labels.append(lbl)

        return labels

    # ── animations ───────────────────────────────────────────────────────────

    def appear_animation(self, run_time: float = DEFAULT_RUN_TIME * 2) -> Succession:
        """
        Boxes materialise top → bottom; arrows grow between them;
        param annotations fade in last.
        """
        box_anim = LaggedStart(
            *[FadeIn(b, shift=RIGHT * 0.12) for b in self._boxes],
            lag_ratio=0.10, run_time=run_time * 0.55,
        )
        arr_anim = LaggedStart(
            *[GrowArrow(a) for a in self._arrows],
            lag_ratio=0.08, run_time=run_time * 0.30,
        )
        param_anim = (
            LaggedStart(
                *[FadeIn(lbl) for lbl in self._param_labels],
                lag_ratio=0.08, run_time=run_time * 0.20,
            )
            if self._param_labels else FadeIn(VGroup())
        )
        return Succession(box_anim, arr_anim, param_anim)

    def flow_animation(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """Data pulse travels top → bottom through every component box."""
        steps: list[Animation] = []
        for box in self._boxes:
            steps.append(
                AnimationGroup(
                    Flash(box._rect, color=COLORS["highlight"],
                          line_length=0.18, run_time=run_time * 0.45),
                    box._rect.animate(run_time=run_time * 0.45)
                              .set_stroke(color=COLORS["highlight"], width=2.5),
                    lag_ratio=0.0,
                )
            )
            steps.append(
                box._rect.animate(run_time=FAST_RUN_TIME * 0.5)
                          .set_stroke(color=WHITE, width=1.1)
            )
        return Succession(*steps)

    def zoom_into(
        self,
        label_fragment: str,
        scale:     float = 2.2,
        run_time:  float = DEFAULT_RUN_TIME,
    ) -> tuple[VGroup | None, AnimationGroup]:
        """
        Scale up the first box whose label contains `label_fragment`.
        Dims all other boxes. Returns (box, animation).
        Call zoom_out() to restore.
        """
        target = next(
            (b for b in self._boxes if label_fragment in b._label.text),
            None,
        )
        if target is None:
            return None, AnimationGroup()

        self._zoom_target = target
        self._zoom_scale  = scale

        anims = [target.animate(run_time=run_time).scale(scale)]
        for box in self._boxes:
            if box is not target:
                anims.append(box.animate(run_time=run_time).set_opacity(0.15))
        for arr in self._arrows:
            anims.append(arr.animate(run_time=run_time).set_opacity(0.10))

        return target, AnimationGroup(*anims, lag_ratio=0.0)

    def zoom_out(self, run_time: float = DEFAULT_RUN_TIME) -> AnimationGroup:
        """Restore after zoom_into."""
        if not hasattr(self, "_zoom_target"):
            return AnimationGroup()
        anims = [
            self._zoom_target.animate(run_time=run_time).scale(1 / self._zoom_scale)
        ]
        for box in self._boxes:
            anims.append(box.animate(run_time=run_time).set_opacity(1.0))
        for arr in self._arrows:
            anims.append(arr.animate(run_time=run_time).set_opacity(1.0))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def highlight_component(
        self,
        label_fragment: str,
        color: str = COLORS["highlight"],
        run_time: float = DEFAULT_RUN_TIME,
    ) -> AnimationGroup:
        """Highlight one box by label fragment; dim all others."""
        anims = []
        for box in self._boxes:
            if label_fragment in box._label.text:
                anims.append(Circumscribe(box, color=color, run_time=run_time))
                anims.append(box.animate(run_time=run_time).set_opacity(1.0))
            else:
                anims.append(box.animate(run_time=run_time).set_opacity(0.18))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def reset_opacity(self, run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        anims = [b.animate(run_time=run_time).set_opacity(1.0) for b in self._boxes]
        anims += [a.animate(run_time=run_time).set_opacity(1.0) for a in self._arrows]
        return AnimationGroup(*anims, lag_ratio=0.0)


# ===========================================================================
# 2.  EndToEndAnim
# ===========================================================================

class EndToEndAnim(VGroup):
    """
    A single complete forward pass narrated step by step:

        1. Input text appears
        2. Tokenisation (words split into token boxes)
        3. Embedding lookup (token → vector)
        4. Positional encoding addition
        5. N Transformer layers (compact stack lights up)
        6. Final hidden state → LM Head
        7. Logit bar chart appears
        8. Token sampled → output text

    The class is self-contained: it builds a compact vertical/horizontal
    scene from lighter-weight inline objects so it fits on one screen
    without importing every heavy sublayer asset.  If sibling modules are
    available it uses them; otherwise it falls back to simple placeholder boxes.

    Parameters
    ----------
    input_text  : str        — the prompt to show
    output_token: str        — the token to sample at the end
    n_layers    : int        — number of blocks in the stack animation
    style       : str        — 'gpt' | 'bert'
    """

    def __init__(
        self,
        input_text:   str = "The cat sat",
        output_token: str = "on",
        n_layers:     int = 6,
        style:        str = "gpt",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._input_text   = input_text
        self._output_token = output_token
        self._n_layers     = n_layers
        self._style        = style

        tokens = input_text.split()

        BOX_W    = 3.50
        BOX_H    = 0.48
        BOX_GAP  = 0.30
        TOK_W    = 0.75
        TOK_H    = 0.38
        TOK_GAP  = 0.12

        tok_colors = [
            COLORS["token"], COLORS["embedding"], COLORS["attention"],
            COLORS["ffn"],   COLORS["query"],     COLORS["value"],
        ]

        # ────────────────────────────────────────────────────────────────────
        # Stage 0: input text
        # ────────────────────────────────────────────────────────────────────
        self._input_lbl = Text(f'"{input_text}"', font="Monospace") \
                              .scale(LABEL_SCALE + 0.08) \
                              .set_color(COLORS["token"])

        # ────────────────────────────────────────────────────────────────────
        # Stage 1: token boxes
        # ────────────────────────────────────────────────────────────────────
        self._token_boxes: list[VGroup] = []
        x = 0.0
        for idx, tok in enumerate(tokens):
            color = tok_colors[idx % len(tok_colors)]
            rect  = RoundedRectangle(
                width=TOK_W, height=TOK_H,
                corner_radius=0.06,
                fill_color=color, fill_opacity=0.80,
                stroke_color=WHITE, stroke_width=0.9,
            )
            lbl = Text(tok, font="Monospace") \
                      .scale(LABEL_SCALE - 0.04).set_color(WHITE) \
                      .move_to(rect.get_center())
            box = VGroup(rect, lbl)
            box.move_to(_np.array([x + TOK_W / 2, 0, 0]))
            x += TOK_W + TOK_GAP
            self._token_boxes.append(box)

        self._token_row = VGroup(*self._token_boxes)
        total_tok_w = x - TOK_GAP
        self._token_row.shift(LEFT * total_tok_w / 2)

        # ────────────────────────────────────────────────────────────────────
        # Stage 2–3: embedding + PE pipeline (compact placeholder boxes)
        # ────────────────────────────────────────────────────────────────────
        self._embed_box = _component_box(
            "Embedding Lookup", COLORS["embedding"],
            width=BOX_W, height=BOX_H,
        )
        self._pe_box = _component_box(
            "Positional Encoding  (+)", COLORS["norm"],
            width=BOX_W, height=BOX_H,
        )

        # ────────────────────────────────────────────────────────────────────
        # Stage 4: transformer layer stack (compact)
        # ────────────────────────────────────────────────────────────────────
        self._layer_boxes: list[VGroup] = []
        for k in range(n_layers):
            lbox = _component_box(
                f"Transformer  Block  {k + 1}", COLORS["attention"],
                width=BOX_W, height=BOX_H * 0.80,
                fill_opacity=0.72,
            )
            self._layer_boxes.append(lbox)
        self._layer_stack = VGroup(*self._layer_boxes)

        # ────────────────────────────────────────────────────────────────────
        # Stage 5: LM head + logits
        # ────────────────────────────────────────────────────────────────────
        self._lm_head_box = _component_box(
            "LM Head  (Linear)", COLORS["output"],
            width=BOX_W, height=BOX_H,
        )
        self._logit_bars = self._build_logit_bars(output_token, width=BOX_W)

        # ────────────────────────────────────────────────────────────────────
        # Stage 6: output
        # ────────────────────────────────────────────────────────────────────
        self._output_lbl = Text(f'→  "{output_token}"', font="Monospace") \
                               .scale(LABEL_SCALE + 0.10) \
                               .set_color(COLORS["highlight"])

        # ────────────────────────────────────────────────────────────────────
        # Vertical layout — stack every stage, separated by flow arrows
        # ────────────────────────────────────────────────────────────────────
        all_stages = [
            self._input_lbl,
            self._token_row,
            self._embed_box,
            self._pe_box,
            *self._layer_boxes,
            self._lm_head_box,
            self._logit_bars,
            self._output_lbl,
        ]

        y = 0.0
        for stage in all_stages:
            stage.move_to(_np.array([0.0, -y, 0]))
            y += stage.height + BOX_GAP

        total_h = y - BOX_GAP
        for stage in all_stages:
            stage.shift(UP * total_h / 2)

        # flow arrows between stages (skip logit_bars → output since they overlap)
        self._flow_arrows: list[Arrow] = []
        for a, b in zip(all_stages[:-2], all_stages[1:-1]):
            arr = Arrow(
                a.get_bottom(), b.get_top(),
                buff=0.06, color=COLORS["arrow"],
                stroke_width=1.2,
                max_tip_length_to_length_ratio=0.28,
            )
            self._flow_arrows.append(arr)
            self.add(arr)

        # final sample arrow (logit bars → output label)
        sample_arr = Arrow(
            self._logit_bars.get_bottom(),
            self._output_lbl.get_top(),
            buff=0.06, color=COLORS["highlight"],
            stroke_width=1.4,
            max_tip_length_to_length_ratio=0.28,
        )
        self._flow_arrows.append(sample_arr)
        self.add(sample_arr)

        self.add(*all_stages)
        self._all_stages = all_stages

        # stage-step labels (left margin)
        step_labels = [
            ("① Input",       self._input_lbl),
            ("② Tokenise",    self._token_row),
            ("③ Embed",       self._embed_box),
            ("④ + PE",        self._pe_box),
            ("⑤ Layers",      self._layer_boxes[0]),
            ("⑥ LM Head",     self._lm_head_box),
            ("⑦ Logits",      self._logit_bars),
            ("⑧ Sample",      self._output_lbl),
        ]
        self._step_lbls: list[Text] = []
        for step_str, anchor in step_labels:
            lbl = _txt(step_str, COLORS["dim"], scale=LABEL_SCALE - 0.05)
            lbl.next_to(anchor, LEFT, buff=0.30)
            self._step_lbls.append(lbl)
            self.add(lbl)

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_logit_bars(top_token: str, width: float = 3.5) -> VGroup:
        """
        Mini bar chart of logit probabilities for a handful of candidate tokens.
        The `top_token` always gets the tallest bar.
        """
        candidates = [top_token, "the", "a", "her", "my"]
        rng  = _np.random.default_rng(7)
        raw  = rng.uniform(0.5, 2.0, len(candidates))
        raw[0] = 3.2   # top_token wins
        probs = _np.exp(raw) / _np.sum(_np.exp(raw))

        group   = VGroup()
        bar_w   = width / len(candidates) * 0.65
        max_h   = 0.55
        spacing = width / len(candidates)
        x_start = -width / 2 + spacing / 2

        for idx, (tok, p) in enumerate(zip(candidates, probs)):
            bar_h = max(0.05, float(p) * max_h * len(candidates) * 0.38)
            color = COLORS["highlight"] if idx == 0 else COLORS["embedding"]
            bar   = Rectangle(
                width=bar_w, height=bar_h,
                fill_color=color, fill_opacity=0.85,
                stroke_color=WHITE, stroke_width=0.4,
            )
            bar.move_to(_np.array([x_start + idx * spacing, bar_h / 2, 0]))

            tok_lbl = _txt(tok, color, scale=LABEL_SCALE - 0.08)
            tok_lbl.next_to(bar, DOWN, buff=0.06)
            pct_lbl = _txt(f"{p*100:.0f}%", color, scale=LABEL_SCALE - 0.10)
            pct_lbl.next_to(bar, UP, buff=0.04)
            group.add(bar, tok_lbl, pct_lbl)

        # baseline
        baseline = Line(
            _np.array([-width / 2, 0, 0]),
            _np.array([ width / 2, 0, 0]),
            color=GRAY, stroke_width=0.8,
        )
        group.add(baseline)
        group._bars = group  # self-ref for external access
        return group

    # ── animations ───────────────────────────────────────────────────────────

    def build_animation(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """
        Play every stage in sequence: text → tokens → embed → PE →
        layer-by-layer → LM head → logits → output.
        """
        steps: list[Animation] = []

        # ① input text
        steps.append(Write(self._input_lbl, run_time=run_time * 0.5))

        # ② tokenise — boxes pop in one by one
        steps.append(LaggedStart(
            *[FadeIn(b, shift=UP * 0.10) for b in self._token_boxes],
            lag_ratio=0.15, run_time=run_time,
        ))

        # ③ embed + ④ PE
        steps.append(FadeIn(self._embed_box, shift=DOWN * 0.10, run_time=run_time * 0.5))
        steps.append(FadeIn(self._pe_box,    shift=DOWN * 0.10, run_time=run_time * 0.5))

        # ⑤ layers — light up one by one
        for lbox in self._layer_boxes:
            steps.append(AnimationGroup(
                FadeIn(lbox, run_time=FAST_RUN_TIME),
                Flash(lbox._rect, color=COLORS["attention"],
                      line_length=0.15, run_time=FAST_RUN_TIME),
                lag_ratio=0.0,
            ))

        # ⑥ LM head
        steps.append(FadeIn(self._lm_head_box, run_time=run_time * 0.5))

        # ⑦ logit bars
        steps.append(FadeIn(self._logit_bars, run_time=run_time * 0.6))

        # ⑧ output token
        steps.append(Write(self._output_lbl, run_time=run_time * 0.5))

        return Succession(*steps)

    def highlight_stage(
        self,
        stage_idx: int,
        color:     str   = COLORS["highlight"],
        run_time:  float = DEFAULT_RUN_TIME,
    ) -> AnimationGroup:
        """
        Spotlight one stage (0-indexed) and dim all others.
        Useful for pausing on a specific step during narration.
        """
        anims = []
        for i, stage in enumerate(self._all_stages):
            if i == stage_idx:
                anims.append(Circumscribe(stage, color=color, run_time=run_time))
            else:
                anims.append(stage.animate(run_time=run_time).set_opacity(0.18))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def reset_opacity(self, run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        return AnimationGroup(
            *[s.animate(run_time=run_time).set_opacity(1.0) for s in self._all_stages],
            lag_ratio=0.0,
        )


# ===========================================================================
# 3.  InferenceLoopAnim
# ===========================================================================

class InferenceLoopAnim(VGroup):
    """
    Wraps the forward pass in an autoregressive loop visual.

    Shows:
        • A growing context window bar (token slots filling up)
        • Each loop iteration: new token generated → appended → context shifts
        • KV-cache memory blocks growing on the side
        • Loop-back arrow from output to input

    Parameters
    ----------
    prompt_tokens   : list[str] — initial prompt tokens
    generated_tokens: list[str] — tokens to generate one by one
    max_seq_len     : int       — total context window slots to display
    show_kv_cache   : bool
    """

    def __init__(
        self,
        prompt_tokens:    list[str]  = None,
        generated_tokens: list[str]  = None,
        max_seq_len:      int        = 10,
        show_kv_cache:    bool       = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if prompt_tokens is None:
            prompt_tokens = ["The", "cat", "sat"]
        if generated_tokens is None:
            generated_tokens = ["on", "the", "mat"]

        self._prompt    = prompt_tokens
        self._generated = generated_tokens
        self._max_len   = max_seq_len
        self._show_kv   = show_kv_cache

        TOK_W = 0.68
        TOK_H = 0.40
        TOK_GAP = 0.10

        tok_colors = [
            COLORS["token"], COLORS["embedding"], COLORS["attention"],
            COLORS["ffn"],   COLORS["query"],     COLORS["value"],
            COLORS["norm"],  COLORS["output"],
        ]

        # ── context window bar (max_seq_len slots) ────────────────────────────
        self._slots: list[VGroup] = []
        self._slot_labels: list[Text] = []

        for i in range(max_seq_len):
            slot = RoundedRectangle(
                width=TOK_W, height=TOK_H,
                corner_radius=0.05,
                fill_color=DARK_GRAY, fill_opacity=0.60,
                stroke_color=GRAY, stroke_width=0.8,
            )
            slot.move_to(_np.array([i * (TOK_W + TOK_GAP), 0, 0]))
            self._slots.append(slot)

        slot_row = VGroup(*self._slots)
        total_w  = max_seq_len * TOK_W + (max_seq_len - 1) * TOK_GAP
        slot_row.shift(LEFT * total_w / 2)
        self.add(slot_row)

        # ── token labels inside slots (filled as tokens arrive) ───────────────
        all_tokens   = prompt_tokens + generated_tokens
        self._tok_lbls: list[Text] = []
        for idx, tok in enumerate(all_tokens[:max_seq_len]):
            color = tok_colors[idx % len(tok_colors)]
            lbl   = Text(tok, font="Monospace") \
                        .scale(LABEL_SCALE - 0.06) \
                        .set_color(color) \
                        .move_to(self._slots[idx].get_center())
            lbl.set_opacity(0)   # hidden until revealed
            self._tok_lbls.append(lbl)
            self.add(lbl)

        # ── KV-cache bar (right side, grows with each new token) ──────────────
        if show_kv_cache:
            kv_lbl = _txt("KV Cache", COLORS["value"], scale=LABEL_SCALE)
            kv_lbl.next_to(slot_row, RIGHT, buff=0.55)
            self._kv_blocks: list[Rectangle] = []
            for i in range(len(all_tokens[:max_seq_len])):
                block = Rectangle(
                    width=0.22, height=TOK_H * 0.70,
                    fill_color=COLORS["value"], fill_opacity=0.75,
                    stroke_color=WHITE, stroke_width=0.4,
                )
                block.next_to(
                    kv_lbl, DOWN, buff=0.08 + i * (TOK_H * 0.70 + 0.06)
                )
                block.set_opacity(0)
                self._kv_blocks.append(block)
                self.add(block)
            self.add(kv_lbl)

        # ── forward pass box ──────────────────────────────────────────────────
        self._fwd_box = _component_box(
            "Forward Pass  (N layers)", COLORS["attention"],
            width=3.00, height=0.50,
        )
        self._fwd_box.next_to(slot_row, DOWN, buff=0.55)
        self.add(self._fwd_box)

        # ── loop-back arrow from output back up to context window ─────────────
        loop_start = self._fwd_box.get_right() + RIGHT * 0.10
        loop_end   = slot_row.get_right()      + RIGHT * 0.10
        self._loop_arrow = Arrow(
            loop_start, loop_end,
            buff=0.06, color=COLORS["highlight"],
            stroke_width=1.4,
            max_tip_length_to_length_ratio=0.28,
        )
        self._loop_lbl = _txt("append\nnew token", COLORS["highlight"],
                               scale=LABEL_SCALE - 0.06)
        self._loop_lbl.next_to(self._loop_arrow, RIGHT, buff=0.12)
        self.add(self._loop_arrow, self._loop_lbl)

        # ── generation counter ────────────────────────────────────────────────
        self._counter = _txt("", COLORS["highlight"], scale=LABEL_SCALE)
        self._counter.next_to(self._fwd_box, DOWN, buff=0.30)
        self.add(self._counter)

        # ── labels ────────────────────────────────────────────────────────────
        ctx_lbl = _txt(f"Context window  ({max_seq_len} tokens max)",
                        COLORS["dim"], scale=LABEL_SCALE - 0.04)
        ctx_lbl.next_to(slot_row, UP, buff=0.22)
        self.add(ctx_lbl)

    # ── animations ───────────────────────────────────────────────────────────

    def appear_animation(self, run_time: float = DEFAULT_RUN_TIME) -> Succession:
        """Context window slots + fwd box + loop arrow appear."""
        return Succession(
            LaggedStart(
                *[FadeIn(s, scale=0.85) for s in self._slots],
                lag_ratio=0.05, run_time=run_time * 0.5,
            ),
            FadeIn(self._fwd_box, run_time=run_time * 0.3),
            GrowArrow(self._loop_arrow, run_time=run_time * 0.3),
            FadeIn(self._loop_lbl, run_time=FAST_RUN_TIME),
        )

    def fill_prompt(self, run_time: float = DEFAULT_RUN_TIME) -> LaggedStart:
        """Reveal prompt tokens one by one in the context window slots."""
        anims = []
        for i in range(min(len(self._prompt), len(self._tok_lbls))):
            anims.append(AnimationGroup(
                self._slots[i].animate(run_time=FAST_RUN_TIME)
                              .set_fill(color=COLORS["token"], opacity=0.65),
                self._tok_lbls[i].animate(run_time=FAST_RUN_TIME).set_opacity(1.0),
                lag_ratio=0.0,
            ))
        return LaggedStart(*anims, lag_ratio=0.18, run_time=run_time)

    def generate_step(
        self,
        step_idx:  int,
        run_time:  float = DEFAULT_RUN_TIME,
    ) -> Succession:
        """
        Animate one autoregressive step:
          forward-pass flash → new token appears in next slot → KV cache grows.
        """
        tok_idx = len(self._prompt) + step_idx
        if tok_idx >= len(self._tok_lbls) or tok_idx >= self._max_len:
            return Succession(FadeIn(VGroup()))   # no-op

        new_tok = (self._generated + ["…"])[step_idx]

        # update counter label
        new_counter = _txt(
            f"Generating token {step_idx + 1} / {len(self._generated)} …",
            COLORS["highlight"], scale=LABEL_SCALE,
        ).move_to(self._counter.get_center())

        fwd_flash = AnimationGroup(
            Flash(self._fwd_box._rect, color=COLORS["attention"],
                  line_length=0.18, run_time=run_time * 0.4),
            self._fwd_box._rect.animate(run_time=run_time * 0.4)
                               .set_stroke(color=COLORS["highlight"], width=2.5),
            lag_ratio=0.0,
        )
        restore_fwd = self._fwd_box._rect.animate(run_time=FAST_RUN_TIME * 0.5) \
                                          .set_stroke(color=WHITE, width=1.1)

        tok_appear = AnimationGroup(
            self._slots[tok_idx].animate(run_time=FAST_RUN_TIME)
                                .set_fill(color=COLORS["highlight"], opacity=0.80),
            self._tok_lbls[tok_idx].animate(run_time=FAST_RUN_TIME)
                                   .set_opacity(1.0),
            lag_ratio=0.0,
        )

        steps: list[Animation] = [
            ReplacementTransform(self._counter, new_counter, run_time=FAST_RUN_TIME),
            fwd_flash,
            restore_fwd,
            tok_appear,
        ]

        if self._show_kv and tok_idx < len(self._kv_blocks):
            steps.append(
                self._kv_blocks[tok_idx].animate(run_time=FAST_RUN_TIME)
                                         .set_opacity(1.0)
            )

        self._counter = new_counter
        self.add(new_counter)
        return Succession(*steps)

    def full_generation_animation(
        self,
        run_time_per_step: float = DEFAULT_RUN_TIME * 0.9,
    ) -> Succession:
        """
        Fill prompt → generate every token in sequence.
        One call gives you the complete autoregressive loop.
        """
        steps: list[Animation] = [self.fill_prompt(run_time_per_step)]
        for i in range(len(self._generated)):
            steps.append(self.generate_step(i, run_time_per_step))
        return Succession(*steps)


# ===========================================================================
# 4.  ModelComparisonTable
# ===========================================================================

class ModelComparisonTable(VGroup):
    """
    Animated comparison table for well-known Transformer models.

    Rows materialise one by one; cells highlight on demand.

    Parameters
    ----------
    models    : list[dict] — override the default model specs
    col_width : float
    row_height: float
    """

    _DEFAULT_MODELS = [
        {"name": "GPT-2 Small", "params": "117 M", "layers": 12,
         "d_model": 768,  "heads": 12, "d_ff": 3072,  "context": 1024},
        {"name": "GPT-2 Large", "params": "774 M", "layers": 36,
         "d_model": 1280, "heads": 20, "d_ff": 5120,  "context": 1024},
        {"name": "GPT-3",       "params": "175 B", "layers": 96,
         "d_model": 12288,"heads": 96, "d_ff": 49152, "context": 2048},
        {"name": "LLaMA-3 8B",  "params": "8 B",   "layers": 32,
         "d_model": 4096, "heads": 32, "d_ff": 14336, "context": 8192},
        {"name": "LLaMA-3 70B", "params": "70 B",  "layers": 80,
         "d_model": 8192, "heads": 64, "d_ff": 28672, "context": 8192},
        {"name": "BERT-Base",   "params": "110 M",  "layers": 12,
         "d_model": 768,  "heads": 12, "d_ff": 3072,  "context": 512},
        {"name": "BERT-Large",  "params": "340 M",  "layers": 24,
         "d_model": 1024, "heads": 16, "d_ff": 4096,  "context": 512},
    ]

    _COLUMNS = ["Model", "Params", "Layers", "d_model", "Heads", "d_ff", "Context"]
    _COL_KEYS = ["name", "params", "layers", "d_model", "heads", "d_ff", "context"]
    _COL_COLORS = [
        COLORS["token"], COLORS["highlight"], COLORS["attention"],
        COLORS["embedding"], COLORS["query"], COLORS["ffn"], COLORS["norm"],
    ]

    def __init__(
        self,
        models:     list[dict] | None = None,
        col_width:  float = 1.15,
        row_height: float = 0.38,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._models     = models or self._DEFAULT_MODELS
        self._col_width  = col_width
        self._row_height = row_height

        self._header_cells: list[VGroup] = []
        self._row_groups:   list[VGroup] = []
        self._all_cells:    list[list[VGroup]] = []

        n_cols = len(self._COLUMNS)
        n_rows = len(self._models)

        # ── header row ────────────────────────────────────────────────────────
        header = VGroup()
        for c_idx, col_name in enumerate(self._COLUMNS):
            cell = self._make_cell(col_name, DARK_GRAY, self._COL_COLORS[c_idx],
                                   col_width, row_height, bold=True)
            cell.move_to(_np.array([c_idx * col_width, 0, 0]))
            header.add(cell)
            self._header_cells.append(cell)
        self.add(header)

        # ── data rows ─────────────────────────────────────────────────────────
        for r_idx, model in enumerate(self._models):
            row    = VGroup()
            row_cells = []
            y_pos  = -(r_idx + 1) * row_height

            for c_idx, key in enumerate(self._COL_KEYS):
                val   = str(model.get(key, "—"))
                color = self._COL_COLORS[c_idx] if c_idx == 0 else LIGHT_GRAY
                bg    = DARK_GRAY if r_idx % 2 == 0 else "#1a1a1a"
                cell  = self._make_cell(val, bg, color, col_width, row_height)
                cell.move_to(_np.array([c_idx * col_width, y_pos, 0]))
                row.add(cell)
                row_cells.append(cell)

            row.set_opacity(0)   # hidden until animated in
            self._row_groups.append(row)
            self._all_cells.append(row_cells)
            self.add(row)

        # centre the whole table
        total_w = n_cols * col_width
        self.shift(LEFT * total_w / 2 + RIGHT * col_width / 2)

        # ── title ─────────────────────────────────────────────────────────────
        title = _txt("Model Comparison", COLORS["highlight"],
                     scale=LABEL_SCALE + 0.08)
        title.next_to(self, UP, buff=0.38)
        self.add(title)

    # ── internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_cell(
        text:       str,
        bg_color:   str,
        text_color: str,
        width:      float,
        height:     float,
        bold:       bool = False,
    ) -> VGroup:
        rect = Rectangle(
            width=width, height=height,
            fill_color=bg_color, fill_opacity=0.90,
            stroke_color=GRAY, stroke_width=0.4,
        )
        scale = LABEL_SCALE - (0.0 if bold else 0.04)
        lbl   = Text(text, font="Monospace").scale(scale).set_color(text_color)
        lbl.move_to(rect.get_center())
        return VGroup(rect, lbl)

    # ── animations ───────────────────────────────────────────────────────────

    def appear_animation(self, run_time: float = DEFAULT_RUN_TIME * 2) -> Succession:
        """Header appears; then rows materialise one by one."""
        header_anim = LaggedStart(
            *[FadeIn(c, shift=DOWN * 0.08) for c in self._header_cells],
            lag_ratio=0.08, run_time=run_time * 0.25,
        )
        row_anim = LaggedStart(
            *[row.animate(run_time=run_time / len(self._row_groups) * 1.2)
                 .set_opacity(1.0)
              for row in self._row_groups],
            lag_ratio=0.15,
            run_time=run_time * 0.75,
        )
        return Succession(header_anim, row_anim)

    def highlight_row(
        self,
        row_idx:  int,
        color:    str   = COLORS["highlight"],
        run_time: float = FAST_RUN_TIME,
    ) -> AnimationGroup:
        """Highlight one model row; dim all others."""
        anims = []
        for r, row in enumerate(self._row_groups):
            if r == row_idx:
                anims.append(row.animate(run_time=run_time).set_opacity(1.0))
                for cell in self._all_cells[r]:
                    anims.append(
                        cell.submobjects[0].animate(run_time=run_time)
                            .set_stroke(color=color, width=1.8)
                    )
            else:
                anims.append(row.animate(run_time=run_time).set_opacity(0.15))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def highlight_col(
        self,
        col_idx:  int,
        run_time: float = FAST_RUN_TIME,
    ) -> AnimationGroup:
        """Highlight one column across all rows."""
        color = self._COL_COLORS[col_idx]
        anims = []
        for r_idx, row_cells in enumerate(self._all_cells):
            for c_idx, cell in enumerate(row_cells):
                if c_idx == col_idx:
                    anims.append(Indicate(cell, color=color, run_time=run_time))
                else:
                    anims.append(cell.animate(run_time=run_time).set_opacity(0.20))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def reset_opacity(self, run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        anims = [row.animate(run_time=run_time).set_opacity(1.0)
                 for row in self._row_groups]
        for row_cells in self._all_cells:
            for cell in row_cells:
                anims.append(cell.animate(run_time=run_time).set_opacity(1.0))
        return AnimationGroup(*anims, lag_ratio=0.0)


# ===========================================================================
# 5.  ArchitectureSelector
# ===========================================================================

class ArchitectureSelector(VGroup):
    """
    Side-by-side columns for the three major Transformer architecture families:
        Encoder-only   |   Decoder-only   |   Encoder-Decoder

    Each column shows:
        • Architecture name + colour
        • Mini component stack
        • Representative models
        • Best-suited tasks

    Parameters
    ----------
    col_width  : float — width of each column
    col_height : float — height of each mini stack
    """

    _ARCH_SPECS = {
        "Encoder-only": {
            "color":  COLORS["embedding"],
            "models": "BERT, RoBERTa,\nDeBERTa, ALBERT",
            "tasks":  "Classification\nNER / POS tagging\nSentence embeddings\nQ&A (extractive)",
            "stack":  [
                ("Token + Pos Embed", COLORS["embedding"]),
                ("Encoder Block × N",  COLORS["attention"]),
                ("Pooler / [CLS]",     COLORS["norm"]),
                ("Task Head",          COLORS["output"]),
            ],
        },
        "Decoder-only": {
            "color":  COLORS["attention"],
            "models": "GPT-2/3/4,\nLLaMA, Mistral,\nGemini, Claude",
            "tasks":  "Text generation\nChat / instruction\nCode generation\nFew-shot learning",
            "stack":  [
                ("Token + Pos Embed",    COLORS["embedding"]),
                ("Causal Block × N",      COLORS["attention"]),
                ("LM Head (Linear)",      COLORS["output"]),
                ("Softmax + Sample",      COLORS["output"]),
            ],
        },
        "Encoder-Decoder": {
            "color":  COLORS["key"],
            "models": "T5, BART,\nmT5, FLAN-T5",
            "tasks":  "Translation\nSummarisation\nQ&A (generative)\nParaphrase",
            "stack":  [
                ("Encoder Embed",         COLORS["embedding"]),
                ("Encoder Block × N",     COLORS["key"]),
                ("Decoder Embed",         COLORS["embedding"]),
                ("Decoder Block × N",     COLORS["attention"]),
                ("LM Head",               COLORS["output"]),
            ],
        },
    }

    def __init__(
        self,
        col_width:  float = 3.00,
        col_height: float = 0.38,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._columns: list[VGroup] = []
        self._col_boxes: list[list[VGroup]] = []
        names = list(self._ARCH_SPECS.keys())

        COL_GAP = 0.45
        total_w = len(names) * col_width + (len(names) - 1) * COL_GAP
        x_start = -total_w / 2 + col_width / 2

        for col_idx, name in enumerate(names):
            spec  = self._ARCH_SPECS[name]
            color = spec["color"]
            x_pos = x_start + col_idx * (col_width + COL_GAP)

            col_group = VGroup()
            col_item_boxes: list[VGroup] = []

            # ── column header ────────────────────────────────────────────────
            header = RoundedRectangle(
                width=col_width, height=col_height * 1.15,
                corner_radius=0.07,
                fill_color=color, fill_opacity=0.85,
                stroke_color=WHITE, stroke_width=1.2,
            )
            header_lbl = Text(name, font="Monospace") \
                             .scale(LABEL_SCALE).set_color(WHITE) \
                             .move_to(header.get_center())
            header_group = VGroup(header, header_lbl)
            col_group.add(header_group)
            col_item_boxes.append(header_group)

            y_cursor = -(col_height * 1.15 + 0.20)

            # ── mini component stack ─────────────────────────────────────────
            for lbl, c in spec["stack"]:
                box = RoundedRectangle(
                    width=col_width * 0.88, height=col_height,
                    corner_radius=0.05,
                    fill_color=c, fill_opacity=0.65,
                    stroke_color=WHITE, stroke_width=0.7,
                )
                box_lbl = Text(lbl, font="Monospace") \
                              .scale(LABEL_SCALE - 0.07).set_color(WHITE) \
                              .move_to(box.get_center())
                bgroup = VGroup(box, box_lbl)
                bgroup.move_to(_np.array([0, y_cursor - col_height / 2, 0]))
                col_group.add(bgroup)
                col_item_boxes.append(bgroup)

                # connecting arrow between stack items
                if len(col_item_boxes) > 2:
                    prev = col_item_boxes[-2]
                    arr  = Arrow(
                        prev.get_bottom(), bgroup.get_top(),
                        buff=0.04, color=GRAY, stroke_width=0.8,
                        max_tip_length_to_length_ratio=0.35,
                    )
                    col_group.add(arr)

                y_cursor -= col_height + 0.12

            # ── representative models ─────────────────────────────────────────
            models_lbl = Text(spec["models"], font="Monospace") \
                             .scale(LABEL_SCALE - 0.06).set_color(color)
            models_lbl.move_to(_np.array([0, y_cursor - 0.30, 0]))
            y_cursor -= models_lbl.height + 0.35

            models_title = _txt("Models:", color, scale=LABEL_SCALE - 0.04)
            models_title.next_to(models_lbl, UP, buff=0.06)

            col_group.add(models_title, models_lbl)

            # ── task list ────────────────────────────────────────────────────
            tasks_lbl = Text(spec["tasks"], font="Monospace") \
                            .scale(LABEL_SCALE - 0.06).set_color(LIGHT_GRAY)
            tasks_lbl.move_to(_np.array([0, y_cursor - 0.20, 0]))

            tasks_title = _txt("Best for:", LIGHT_GRAY, scale=LABEL_SCALE - 0.04)
            tasks_title.next_to(tasks_lbl, UP, buff=0.06)

            col_group.add(tasks_title, tasks_lbl)

            # ── position column ───────────────────────────────────────────────
            col_group.move_to(_np.array([x_pos, 0, 0]))
            self._columns.append(col_group)
            self._col_boxes.append(col_item_boxes)
            self.add(col_group)

        # vertical separator lines between columns
        for i in range(len(names) - 1):
            x_sep = x_start + (i + 1) * (col_width + COL_GAP) - COL_GAP / 2
            sep   = DashedLine(
                _np.array([x_sep,  1.5, 0]),
                _np.array([x_sep, -4.5, 0]),
                color=COLORS["dim"], stroke_width=0.7, dash_length=0.10,
            )
            self.add(sep)

        # title
        title = _txt("Transformer Architecture Families",
                     COLORS["highlight"], scale=LABEL_SCALE + 0.08)
        title.next_to(self, UP, buff=0.45)
        self.add(title)

    # ── animations ───────────────────────────────────────────────────────────

    def appear_animation(self, run_time: float = DEFAULT_RUN_TIME * 2) -> LaggedStart:
        """All three columns fade in side by side, slightly staggered."""
        return LaggedStart(
            *[FadeIn(col, shift=UP * 0.15) for col in self._columns],
            lag_ratio=0.25,
            run_time=run_time,
        )

    def highlight_column(
        self,
        col_idx:  int,
        run_time: float = DEFAULT_RUN_TIME,
    ) -> AnimationGroup:
        """Spotlight one architecture column; dim the other two."""
        anims = []
        for i, col in enumerate(self._columns):
            if i == col_idx:
                anims.append(col.animate(run_time=run_time).set_opacity(1.0))
            else:
                anims.append(col.animate(run_time=run_time).set_opacity(0.15))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def highlight_by_name(
        self,
        name_fragment: str,
        run_time:      float = DEFAULT_RUN_TIME,
    ) -> AnimationGroup:
        """Highlight by architecture name fragment e.g. 'Decoder'."""
        names = list(self._ARCH_SPECS.keys())
        idx   = next(
            (i for i, n in enumerate(names) if name_fragment in n), None
        )
        if idx is None:
            return AnimationGroup()
        return self.highlight_column(idx, run_time)

    def reset_opacity(self, run_time: float = FAST_RUN_TIME) -> AnimationGroup:
        return AnimationGroup(
            *[col.animate(run_time=run_time).set_opacity(1.0)
              for col in self._columns],
            lag_ratio=0.0,
        )

    def annotate_task(
        self,
        col_idx:  int,
        task:     str,
        run_time: float = DEFAULT_RUN_TIME,
    ) -> tuple[VGroup, Succession]:
        """
        Show a highlighted task annotation bubble next to column `col_idx`.
        Returns (bubble_group, animation).
        """
        col = self._columns[col_idx]
        bubble = RoundedRectangle(
            width=2.2, height=0.45,
            corner_radius=0.08,
            fill_color=COLORS["highlight"], fill_opacity=0.90,
            stroke_width=0,
        )
        bubble_lbl = Text(task, font="Monospace") \
                         .scale(LABEL_SCALE).set_color(BLACK) \
                         .move_to(bubble.get_center())
        group = VGroup(bubble, bubble_lbl)
        group.next_to(col, RIGHT, buff=0.35)

        return group, Succession(
            FadeIn(group, shift=LEFT * 0.15, run_time=run_time * 0.5),
            Flash(bubble, color=COLORS["highlight"],
                  line_length=0.16, run_time=run_time * 0.5),
        )