"""
manim_transformer/output/logits_sampling.py
=============================================
Reusable Manim assets for the OUTPUT stage of a Transformer:

    LogitsBar          — animated bar chart of raw logit / probability scores
    TemperatureAnim    — live temperature slider morphing a distribution
    TopKFilter         — visually zeros out below-top-K bars
    TopPFilter         — cumulative-sum nucleus sampling cutoff
    TokenSelectionAnim — samples one token, appends it to the output sequence
    KVCacheAnim        — shows KV cache slots filling as tokens are generated

Design principles
-----------------
* Every class extends VGroup → composable, positionable, animatable.
* Every class exposes an ``animate_*`` method that returns a list of
  Manim Animation objects, ready to be passed to ``self.play(*anims)``.
* All magic numbers come from styles/constants.py; all colors from
  styles/colors.py; all text from styles/fonts.py.

Usage example
-------------
    from manim import *
    from manim_transformer.output.logits_sampling import (
        LogitsBar, TemperatureAnim, TopKFilter,
        TopPFilter, TokenSelectionAnim, KVCacheAnim,
    )

    class SamplingDemo(Scene):
        def construct(self):
            import torch, torch.nn.functional as F

            logits = [3.1, 2.8, 1.2, 0.9, 0.4, -0.2, -0.8, -1.5,
                      -2.0, -2.4, -2.9, -3.1]
            tokens = ["the","a","an","its","their","his","her",
                      "our","your","my","this","that"]

            bar = LogitsBar(logits, tokens).shift(UP)
            self.play(*bar.animate_appear())
            self.play(*bar.animate_softmax())
            self.wait()
"""

from __future__ import annotations

import math
from typing import Sequence

from manim import (
    VGroup, Rectangle, Text, Line, Arrow, Dot,
    FadeIn, FadeOut, Transform, Create, Write,
    GrowFromEdge, AnimationGroup, Succession,
    always_redraw, ValueTracker,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    WHITE, YELLOW, GREEN, RED, GRAY,
    ManimColor, Axes, DecimalNumber,
    RoundedRectangle, Brace, DashedLine,
)

from manim_transformer.styles.colors import (
    UI_COLORS, SAMPLING_COLORS, COMPONENT_COLORS, TOKEN_PALETTE,
)
from manim_transformer.styles.constants import SAMPLING, TIMING, TOKEN, TYPOGRAPHY as TY
from manim_transformer.styles.fonts import (
    make_text, STYLE, label, caption, small, token_word, callout,
)


# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------

def _softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    scaled = [x / max(temperature, 1e-9) for x in logits]
    max_val = max(scaled)
    exps = [math.exp(x - max_val) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def _normalize(probs: list[float]) -> list[float]:
    total = sum(probs)
    if total < 1e-9:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in probs]


def _bar_color(prob: float) -> ManimColor:
    """High prob → yellow/green, low → dim grey."""
    if prob > 0.30:
        return SAMPLING_COLORS["selected"]
    elif prob > 0.10:
        return SAMPLING_COLORS["top_k"]
    elif prob > 0.01:
        return UI_COLORS["label_dim"]
    else:
        return SAMPLING_COLORS["top_k_cut"]


# ---------------------------------------------------------------------------
# 1. LogitsBar
# ---------------------------------------------------------------------------

class LogitsBar(VGroup):
    """
    Animated bar chart displaying raw logits OR softmax probabilities.

    Layout
    ------
    Bars grow upward from a baseline.  Token labels sit below each bar.
    Probability / logit values optionally shown above bar tops.

    Parameters
    ----------
    logits : list[float]
        Raw logit values (one per token).
    tokens : list[str]
        Token strings matching ``logits`` length.
    max_display : int
        Cap number of bars shown (highest-logit tokens kept).
    show_values : bool
        Show numeric label above each bar.
    bar_width : float
        Bar width in Manim units.
    max_height : float
        Maximum bar height in Manim units.

    Key methods
    -----------
    animate_appear()   → list[Animation]   Bars grow from baseline
    animate_softmax()  → list[Animation]   Rescale heights to probs
    animate_highlight(index)               Glow one bar
    get_probs()        → list[float]       Current probability values
    """

    def __init__(
        self,
        logits: list[float],
        tokens: list[str],
        max_display: int = SAMPLING.MAX_BARS_DISPLAY,
        show_values: bool = True,
        bar_width: float = SAMPLING.LOGIT_BAR_WIDTH,
        max_height: float = SAMPLING.LOGIT_BAR_MAX_H,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # --- Clamp to top-N by logit value -----------------------------------
        paired = sorted(zip(logits, tokens), reverse=True)[:max_display]
        self._logits = [p[0] for p in paired]
        self._tokens = [p[1] for p in paired]
        self._probs  = _softmax(self._logits)
        self._n      = len(self._logits)

        self._bar_width  = bar_width
        self._max_height = max_height
        self._spacing    = SAMPLING.LOGIT_BAR_SPACING
        self._show_values = show_values

        # --- Normalise heights (logit mode: shift to 0-based) ----------------
        min_l = min(self._logits)
        max_l = max(self._logits)
        span  = max(max_l - min_l, 1e-9)
        self._logit_heights = [
            max_height * (l - min_l) / span for l in self._logits
        ]
        self._prob_heights = [
            max_height * p / max(self._probs) for p in self._probs
        ]

        # --- Build initial bars (logit heights) ------------------------------
        self._bars:        list[Rectangle] = []
        self._bar_labels:  list[Text]      = []
        self._val_labels:  list[Text]      = []
        self._baseline     = Line(LEFT * 0.3, RIGHT * 0.3, color=UI_COLORS["grid_line"])

        total_w = self._n * (bar_width + self._spacing) - self._spacing
        x_start = -total_w / 2 + bar_width / 2

        for i, (h, tok) in enumerate(zip(self._logit_heights, self._tokens)):
            x = x_start + i * (bar_width + self._spacing)

            # Bar rectangle
            bar = Rectangle(
                width=bar_width,
                height=max(h, 0.02),
                fill_color=_bar_color(self._probs[i]),
                fill_opacity=0.85,
                stroke_color=UI_COLORS["border"],
                stroke_width=TY.STROKE_THIN,
            )
            bar.move_to([x, max(h, 0.02) / 2, 0])   # anchor at y=0 baseline
            self._bars.append(bar)

            # Token label below bar
            tok_lbl = make_text(
                tok[:8],      # truncate long tokens
                style=STYLE.SMALL,
                color=UI_COLORS["label_dim"],
            )
            tok_lbl.move_to([x, -0.28, 0])
            self._bar_labels.append(tok_lbl)

            # Value label above bar
            if show_values:
                val_str = f"{self._logits[i]:.2f}"
                val_lbl = make_text(val_str, style=STYLE.MATRIX_VAL, color=UI_COLORS["label_dim"])
                val_lbl.move_to([x, max(h, 0.02) + 0.18, 0])
                self._val_labels.append(val_lbl)

        # Baseline spanning all bars
        self._baseline = Line(
            [-total_w / 2, 0, 0],
            [ total_w / 2, 0, 0],
            color=UI_COLORS["grid_line"],
            stroke_width=1.5,
        )

        # Add everything
        self.add(self._baseline)
        for b in self._bars:
            self.add(b)
        for lbl in self._bar_labels:
            self.add(lbl)
        if show_values:
            for v in self._val_labels:
                self.add(v)

        # State flag
        self._showing_probs = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_probs(self) -> list[float]:
        return list(self._probs)

    def get_tokens(self) -> list[str]:
        return list(self._tokens)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_appear(self) -> list:
        """
        Bars grow upward from the baseline one by one.

        Returns a list of animations to pass to ``self.play()``.

        Example
        -------
        >>> self.play(*bar_chart.animate_appear())
        """
        anims = [FadeIn(self._baseline)]
        for i, (bar, lbl) in enumerate(zip(self._bars, self._bar_labels)):
            anims.append(GrowFromEdge(bar, DOWN))
            anims.append(FadeIn(lbl))
        if self._show_values:
            for v in self._val_labels:
                anims.append(FadeIn(v))
        return anims

    def animate_softmax(self, temperature: float = 1.0) -> list:
        """
        Rescale all bar heights from raw-logit proportions to
        softmax probability proportions.

        Simultaneously updates value labels to show ``p=0.xx``.

        Parameters
        ----------
        temperature : float
            Softmax temperature (default 1.0).

        Example
        -------
        >>> self.play(*bar_chart.animate_softmax(), run_time=1.2)
        """
        self._probs = _softmax(self._logits, temperature)
        self._prob_heights = [
            self._max_height * p / max(self._probs) for p in self._probs
        ]
        anims = []
        for i, bar in enumerate(self._bars):
            new_h = max(self._prob_heights[i], 0.02)
            new_bar = Rectangle(
                width=self._bar_width,
                height=new_h,
                fill_color=_bar_color(self._probs[i]),
                fill_opacity=0.85,
                stroke_color=UI_COLORS["border"],
                stroke_width=TY.STROKE_THIN,
            )
            x = bar.get_center()[0]
            new_bar.move_to([x, new_h / 2, 0])
            anims.append(Transform(bar, new_bar, run_time=TIMING.SOFTMAX))

            if self._show_values:
                new_val = make_text(
                    f"p={self._probs[i]:.3f}",
                    style=STYLE.MATRIX_VAL,
                    color=UI_COLORS["label_dim"],
                )
                new_val.move_to([x, new_h + 0.18, 0])
                anims.append(Transform(self._val_labels[i], new_val, run_time=TIMING.SOFTMAX))

        self._showing_probs = True
        return anims

    def animate_highlight(self, index: int, color: ManimColor | None = None) -> list:
        """
        Glow-highlight a single bar (e.g. the sampled token).

        Parameters
        ----------
        index : int
            Bar index to highlight (0-based, within displayed bars).
        color : ManimColor, optional
            Override highlight color (default: ``SAMPLING_COLORS["selected"]``).

        Example
        -------
        >>> self.play(*bar_chart.animate_highlight(0))
        """
        color = color or SAMPLING_COLORS["selected"]
        bar = self._bars[index]
        new_bar = bar.copy().set_fill(color, opacity=1.0).set_stroke(color, width=3.0)
        return [Transform(bar, new_bar, run_time=TIMING.HIGHLIGHT)]

    def animate_dim_except(self, keep_indices: list[int]) -> list:
        """
        Dim all bars except those at ``keep_indices``.

        Used by TopKFilter and TopPFilter to grey out eliminated tokens.

        Example
        -------
        >>> self.play(*bar_chart.animate_dim_except([0, 1, 2]))
        """
        anims = []
        for i, bar in enumerate(self._bars):
            if i not in keep_indices:
                new_bar = bar.copy().set_fill(
                    SAMPLING_COLORS["top_k_cut"], opacity=0.2
                ).set_stroke(UI_COLORS["dim"], width=0.5)
                anims.append(Transform(bar, new_bar, run_time=TIMING.FAST))
        return anims

    def animate_restore_all(self) -> list:
        """Un-dim all bars back to their probability colours."""
        anims = []
        for i, bar in enumerate(self._bars):
            new_bar = bar.copy().set_fill(
                _bar_color(self._probs[i]), opacity=0.85
            ).set_stroke(UI_COLORS["border"], width=TY.STROKE_THIN)
            anims.append(Transform(bar, new_bar, run_time=TIMING.FAST))
        return anims


# ---------------------------------------------------------------------------
# 2. TemperatureAnim
# ---------------------------------------------------------------------------

class TemperatureAnim(VGroup):
    """
    A ``LogitsBar`` paired with a temperature slider.

    Moving the slider (via a ``ValueTracker``) morphs the bar chart
    between a peaked distribution (T→0) and a flat one (T→∞).

    Parameters
    ----------
    logits : list[float]
    tokens : list[str]
    t_min : float   — left end of slider (default 0.1)
    t_max : float   — right end of slider (default 4.0)

    Key methods
    -----------
    animate_set_temperature(t)  → list[Animation]   Snap to temperature t
    animate_sweep(t_start, t_end, steps) → list     Smooth sweep animation
    get_tracker() → ValueTracker                    For always_redraw usage

    Example
    -------
    >>> ta = TemperatureAnim(logits, tokens)
    >>> self.add(ta)
    >>> self.play(*ta.animate_sweep(1.0, 0.1, steps=20), run_time=3)
    """

    def __init__(
        self,
        logits: list[float],
        tokens: list[str],
        t_min: float = 0.1,
        t_max: float = 4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._logits = logits
        self._tokens = tokens
        self._t_min  = t_min
        self._t_max  = t_max
        self._t_cur  = 1.0

        # Bar chart (starts at T=1.0)
        self._bar = LogitsBar(logits, tokens)
        self._bar.shift(UP * 0.8)

        # Slider track
        self._track_w = 3.5
        self._track = Line(
            LEFT  * self._track_w / 2,
            RIGHT * self._track_w / 2,
            color=UI_COLORS["grid_line"],
            stroke_width=2,
        ).shift(DOWN * 1.6)

        # Slider knob
        self._knob = Dot(radius=0.12, color=SAMPLING_COLORS["temp_warm"])
        self._knob.move_to(self._track.get_center())  # T=1.0 midpoint

        # Temperature label
        self._t_label = make_text("T = 1.00", style=STYLE.LABEL, color=UI_COLORS["highlight"])
        self._t_label.next_to(self._track, DOWN, buff=0.25)

        # Endpoint labels
        cold_lbl = make_text("Cold (sharp)", style=STYLE.SMALL, color=SAMPLING_COLORS["temp_cold"])
        hot_lbl  = make_text("Hot (flat)",   style=STYLE.SMALL, color=SAMPLING_COLORS["temp_hot"])
        cold_lbl.next_to(self._track, LEFT,  buff=0.15)
        hot_lbl.next_to( self._track, RIGHT, buff=0.15)

        self.add(self._bar, self._track, self._knob,
                 self._t_label, cold_lbl, hot_lbl)

    # helpers
    def _t_to_x(self, t: float) -> float:
        """Map temperature value to x-position on the slider track."""
        frac = (t - self._t_min) / (self._t_max - self._t_min)
        frac = max(0.0, min(1.0, frac))
        return self._track.get_left()[0] + frac * self._track_w

    def _knob_color(self, t: float) -> ManimColor:
        if t < 0.8:   return SAMPLING_COLORS["temp_cold"]
        if t < 1.8:   return SAMPLING_COLORS["temp_warm"]
        return SAMPLING_COLORS["temp_hot"]

    def get_tracker(self) -> ValueTracker:
        """Return a ValueTracker for use with ``always_redraw``."""
        return ValueTracker(self._t_cur)

    def animate_set_temperature(self, t: float) -> list:
        """
        Snap to a new temperature value.

        Returns animations:  knob moves + bars rescale + label updates.

        Example
        -------
        >>> self.play(*ta.animate_set_temperature(0.3), run_time=0.8)
        """
        self._t_cur = t
        x_new = self._t_to_x(t)
        y_knob = self._knob.get_center()[1]

        new_knob = Dot(radius=0.12, color=self._knob_color(t))
        new_knob.move_to([x_new, y_knob, 0])

        new_label = make_text(f"T = {t:.2f}", style=STYLE.LABEL, color=UI_COLORS["highlight"])
        new_label.next_to(self._track, DOWN, buff=0.25)

        anims = [
            Transform(self._knob,   new_knob,  run_time=TIMING.FAST),
            Transform(self._t_label, new_label, run_time=TIMING.FAST),
        ]
        anims += self._bar.animate_softmax(temperature=t)
        return anims

    def animate_sweep(
        self,
        t_start: float,
        t_end: float,
        steps: int = 15,
    ) -> list:
        """
        Smooth sweep from ``t_start`` to ``t_end`` in ``steps`` keyframes.

        Returns a ``Succession`` wrapping all step animations.

        Example
        -------
        >>> self.play(ta.animate_sweep(1.0, 0.1, steps=20), run_time=4)
        """
        step_anims = []
        for i in range(steps + 1):
            t = t_start + (t_end - t_start) * i / steps
            step_anims.append(
                AnimationGroup(*self.animate_set_temperature(t))
            )
        return [Succession(*step_anims)]


# ---------------------------------------------------------------------------
# 3. TopKFilter
# ---------------------------------------------------------------------------

class TopKFilter(VGroup):
    """
    Visualise top-K filtering on a ``LogitsBar``.

    Shows which tokens are kept (bright) and which are zeroed out (dim),
    then renormalises the kept distribution.

    Parameters
    ----------
    logits_bar : LogitsBar
        An existing ``LogitsBar`` instance (already added to scene).
    k : int
        Number of top tokens to keep.

    Key methods
    -----------
    animate_filter()      → list[Animation]   Dim the bottom-(N-K) bars
    animate_renormalize() → list[Animation]   Rescale kept bars to sum=1
    animate_full_pass()   → list[Animation]   filter + renormalize together

    Example
    -------
    >>> bars = LogitsBar(logits, tokens)
    >>> self.play(*bars.animate_appear())
    >>> self.play(*bars.animate_softmax())
    >>>
    >>> topk = TopKFilter(bars, k=5)
    >>> self.add(topk)
    >>> self.play(*topk.animate_full_pass())
    """

    def __init__(self, logits_bar: LogitsBar, k: int = SAMPLING.TOP_K_DEFAULT, **kwargs):
        super().__init__(**kwargs)
        self._bar = logits_bar
        self._k   = min(k, len(logits_bar.get_probs()))

        probs = logits_bar.get_probs()
        # Indices sorted by probability (descending)
        sorted_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        self._keep_idx = sorted_idx[:self._k]
        self._cut_idx  = sorted_idx[self._k:]

        # "Top-K = N" banner
        self._banner = make_text(
            f"Top-K  (k = {self._k})",
            style=STYLE.SUBHEADING,
            color=SAMPLING_COLORS["top_k"],
        )

        # Cut-off line (dashed, drawn at the K-th bar level)
        self._cutline: DashedLine | None = None

        self.add(self._banner)

    def animate_filter(self) -> list:
        """
        Dim bars outside top-K, draw a dashed cut-off line.

        Example
        -------
        >>> self.play(*topk.animate_filter(), run_time=0.8)
        """
        anims = self._bar.animate_dim_except(self._keep_idx)

        # Dashed line at the height of the K-th bar
        probs = self._bar.get_probs()
        kth_prob = sorted(probs, reverse=True)[self._k - 1]
        kth_h = self._bar._max_height * kth_prob / max(probs)

        self._cutline = DashedLine(
            self._bar.get_left()  + UP * kth_h,
            self._bar.get_right() + UP * kth_h,
            color=SAMPLING_COLORS["temp_hot"],
            stroke_width=1.5,
            dash_length=0.12,
        )
        cut_lbl = make_text("cut-off", style=STYLE.SMALL, color=SAMPLING_COLORS["temp_hot"])
        cut_lbl.next_to(self._cutline, RIGHT, buff=0.1)
        anims += [Create(self._cutline), FadeIn(cut_lbl)]
        return anims

    def animate_renormalize(self) -> list:
        """
        Rescale the kept bars so their probabilities sum to 1.

        Example
        -------
        >>> self.play(*topk.animate_renormalize(), run_time=0.8)
        """
        probs = self._bar.get_probs()
        kept_probs = [probs[i] if i in self._keep_idx else 0.0
                      for i in range(len(probs))]
        norm_probs = _normalize(kept_probs)

        # Temporarily override _probs so animate_softmax uses new heights
        self._bar._probs = norm_probs
        return self._bar.animate_softmax(temperature=1.0)

    def animate_full_pass(self) -> list:
        """filter + renormalize in sequence."""
        return self.animate_filter() + self.animate_renormalize()


# ---------------------------------------------------------------------------
# 4. TopPFilter
# ---------------------------------------------------------------------------

class TopPFilter(VGroup):
    """
    Visualise nucleus (top-P) sampling on a ``LogitsBar``.

    Bars are sorted descending; a cumulative-sum line sweeps right
    and stops at the nucleus threshold ``p``.

    Parameters
    ----------
    logits_bar : LogitsBar
    p : float   Nucleus probability threshold (default 0.9).

    Key methods
    -----------
    animate_sort()              → list[Animation]   Sort bars descending
    animate_cumsum_line()       → list[Animation]   Draw cumulative-sum line
    animate_mark_nucleus()      → list[Animation]   Highlight nucleus bars
    animate_full_pass()         → list[Animation]   All three steps

    Example
    -------
    >>> topp = TopPFilter(bars, p=0.9)
    >>> self.play(*topp.animate_full_pass())
    """

    def __init__(self, logits_bar: LogitsBar, p: float = SAMPLING.TOP_P_DEFAULT, **kwargs):
        super().__init__(**kwargs)
        self._bar = logits_bar
        self._p   = p

        probs  = logits_bar.get_probs()
        sorted_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        self._sorted_idx = sorted_idx

        # Determine nucleus indices (cumsum ≤ p)
        cumsum = 0.0
        self._nucleus_idx: list[int] = []
        for idx in sorted_idx:
            self._nucleus_idx.append(idx)
            cumsum += probs[idx]
            if cumsum >= p:
                break

        self._cut_idx = [i for i in sorted_idx if i not in self._nucleus_idx]

        banner = make_text(
            f"Top-P  (p = {p:.2f})",
            style=STYLE.SUBHEADING,
            color=SAMPLING_COLORS["top_k"],
        )
        self.add(banner)

    def animate_sort(self) -> list:
        """
        Visually re-sort bars from highest to lowest probability.

        (Moves bar mobjects to new positions — does not mutate internal order.)
        """
        probs = self._bar.get_probs()
        n = len(self._sorted_idx)
        bar_w = self._bar._bar_width
        spacing = self._bar._spacing
        total_w = n * (bar_w + spacing) - spacing
        x_start = -total_w / 2 + bar_w / 2

        anims = []
        for new_pos, old_idx in enumerate(self._sorted_idx):
            bar = self._bar._bars[old_idx]
            x_new = x_start + new_pos * (bar_w + spacing)
            new_bar = bar.copy().move_to([x_new, bar.get_center()[1], 0])
            anims.append(Transform(bar, new_bar, run_time=TIMING.NORMAL))
        return anims

    def animate_cumsum_line(self) -> list:
        """
        Draw a line across the tops of the sorted bars representing
        the cumulative probability sum, and highlight where it crosses p.
        """
        probs   = self._bar.get_probs()
        bar_w   = self._bar._bar_width
        spacing = self._bar._spacing
        max_p   = max(probs)

        bar_xs = []
        cumsum_ys = []
        cum = 0.0
        for idx in self._sorted_idx:
            cum += probs[idx]
            bar_xs.append(idx)
            cumsum_ys.append(min(cum, 1.0))

        # Build polyline points
        n = len(self._sorted_idx)
        total_w = n * (bar_w + spacing) - spacing
        x_start = -total_w / 2 + bar_w / 2
        pts = []
        for j, idx in enumerate(self._sorted_idx):
            x = x_start + j * (bar_w + spacing)
            y = self._bar._max_height * cumsum_ys[j] / 1.0   # 1.0 = max cumsum
            pts.append([x, y, 0])

        # Draw segments one by one
        anims = []
        for i in range(len(pts) - 1):
            seg = Line(pts[i], pts[i + 1], color=SAMPLING_COLORS["temp_warm"], stroke_width=2)
            anims.append(Create(seg, run_time=TIMING.FAST))

        # Horizontal threshold line at p
        thresh_y = self._bar._max_height * self._p
        thresh = DashedLine(
            [-total_w / 2, thresh_y, 0],
            [ total_w / 2, thresh_y, 0],
            color=UI_COLORS["highlight"],
            stroke_width=1.5,
            dash_length=0.12,
        )
        p_lbl = make_text(f"p = {self._p:.2f}", style=STYLE.SMALL, color=UI_COLORS["highlight"])
        p_lbl.next_to(thresh, RIGHT, buff=0.1)
        anims += [Create(thresh), FadeIn(p_lbl)]
        return anims

    def animate_mark_nucleus(self) -> list:
        """Highlight nucleus bars; dim the rest."""
        anims  = self._bar.animate_dim_except(self._nucleus_idx)
        anims += [self._bar.animate_highlight(i)[0] for i in self._nucleus_idx]
        return anims

    def animate_full_pass(self) -> list:
        """Sort → cumsum line → mark nucleus."""
        return self.animate_sort() + self.animate_cumsum_line() + self.animate_mark_nucleus()


# ---------------------------------------------------------------------------
# 5. TokenSelectionAnim
# ---------------------------------------------------------------------------

class TokenSelectionAnim(VGroup):
    """
    Sample one token from a distribution and append it to an output sequence.

    Workflow
    --------
    1. Show a probability distribution (LogitsBar already on screen).
    2. Draw a vertical "sample" arrow dropping onto the winning bar.
    3. The token label lifts off the bar and flies to the output sequence.
    4. Output sequence box updates.

    Parameters
    ----------
    logits_bar : LogitsBar
        The source distribution.
    output_sequence : list[str]
        Tokens already generated (empty list for first token).
    sampled_index : int
        Which bar index was sampled (0-based within displayed bars).

    Key methods
    -----------
    animate_sample_arrow()  → list   Draw arrow onto the winning bar
    animate_token_fly()     → list   Token lifts off and joins sequence
    animate_full()          → list   Both steps together

    Example
    -------
    >>> sel = TokenSelectionAnim(bars, output_sequence=["The"], sampled_index=0)
    >>> self.add(sel)
    >>> self.play(*sel.animate_full())
    """

    def __init__(
        self,
        logits_bar: LogitsBar,
        output_sequence: list[str],
        sampled_index: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._bar      = logits_bar
        self._seq      = list(output_sequence)
        self._sidx     = sampled_index
        self._token    = logits_bar.get_tokens()[sampled_index]

        # --- Output sequence display ----------------------------------------
        self._seq_group = self._build_sequence_display()
        self._seq_group.to_edge(DOWN, buff=0.5)
        self.add(self._seq_group)

        # --- Sample arrow (hidden until animate_sample_arrow) ---------------
        bar_top = self._bar._bars[sampled_index].get_top()
        self._arrow = Arrow(
            bar_top + UP * 0.8,
            bar_top,
            buff=0.05,
            color=UI_COLORS["highlight"],
            stroke_width=3,
            max_tip_length_to_length_ratio=0.25,
        )
        self._arrow.set_opacity(0)
        self.add(self._arrow)

    def _build_sequence_display(self) -> VGroup:
        """Render the current output sequence as a row of token boxes."""
        group = VGroup()
        tokens = self._seq + ["?"]   # "?" placeholder for next token
        for i, tok in enumerate(tokens):
            color = TOKEN_PALETTE[i % len(TOKEN_PALETTE)]
            is_placeholder = (i == len(tokens) - 1)
            box = RoundedRectangle(
                width=TOKEN.BOX_WIDTH,
                height=TOKEN.BOX_HEIGHT,
                corner_radius=TOKEN.BOX_RADIUS,
                fill_color=color if not is_placeholder else UI_COLORS["surface"],
                fill_opacity=0.7 if not is_placeholder else 0.3,
                stroke_color=color,
                stroke_width=TOKEN.BOX_STROKE,
            )
            lbl = make_text(
                tok, style=STYLE.TOKEN_WORD,
                color=WHITE if not is_placeholder else UI_COLORS["dim"],
            )
            lbl.move_to(box.get_center())
            group.add(VGroup(box, lbl))

        group.arrange(RIGHT, buff=TOKEN.SPACING)
        return group

    def animate_sample_arrow(self) -> list:
        """
        Fade in the sample arrow dropping onto the winning bar.

        Example
        -------
        >>> self.play(*sel.animate_sample_arrow(), run_time=0.5)
        """
        return [
            self._arrow.animate.set_opacity(1).run_time(TIMING.ARROW_DRAW),
            *self._bar.animate_highlight(self._sidx),
        ]

    def animate_token_fly(self) -> list:
        """
        The selected token label lifts off the bar and lands
        in the ``?`` placeholder slot of the output sequence.

        Example
        -------
        >>> self.play(*sel.animate_token_fly(), run_time=0.8)
        """
        # Build a flying token box starting at the winning bar
        bar_center = self._bar._bars[self._sidx].get_top() + UP * 0.1
        tok_color  = TOKEN_PALETTE[len(self._seq) % len(TOKEN_PALETTE)]

        fly_box = RoundedRectangle(
            width=TOKEN.BOX_WIDTH, height=TOKEN.BOX_HEIGHT,
            corner_radius=TOKEN.BOX_RADIUS,
            fill_color=tok_color, fill_opacity=0.9,
            stroke_color=tok_color, stroke_width=TOKEN.BOX_STROKE,
        )
        fly_lbl = make_text(self._token, style=STYLE.TOKEN_WORD)
        fly_group = VGroup(fly_box, fly_lbl)
        fly_lbl.move_to(fly_box.get_center())
        fly_group.move_to(bar_center)

        # Target: the "?" placeholder slot
        placeholder = self._seq_group[-1]

        anims = [
            FadeIn(fly_group, run_time=TIMING.FAST),
            fly_group.animate(run_time=TIMING.FLOW_THROUGH).move_to(
                placeholder.get_center()
            ),
            FadeOut(placeholder, run_time=TIMING.FAST),
            self._arrow.animate(run_time=TIMING.FAST).set_opacity(0),
        ]
        return anims

    def animate_full(self) -> list:
        """sample_arrow + token_fly."""
        return self.animate_sample_arrow() + self.animate_token_fly()


# ---------------------------------------------------------------------------
# 6. KVCacheAnim
# ---------------------------------------------------------------------------

class KVCacheAnim(VGroup):
    """
    Visualise the KV cache growing slot-by-slot as new tokens are generated.

    Layout
    ------
    A horizontal row of memory slots.  Each slot shows K and V sub-cells.
    As each new token is generated:
      - Past slots glow briefly (cache HIT).
      - A new slot appears on the right (cache WRITE).
      - A speed comparison shows "with cache vs without" compute savings.

    Parameters
    ----------
    max_seq_len : int
        Maximum context length to display (caps slot count).
    d_k : int
        Key/value head dimension (display only, shown in label).
    num_heads : int
        Number of attention heads (display only).
    show_speed_bar : bool
        Show the compute-saving comparison bar below the cache.

    Key methods
    -----------
    animate_init()                → list   Show empty cache
    animate_cache_hit(step)       → list   Flash previously filled slots
    animate_cache_write(step, tok)→ list   Fill the next empty slot
    animate_step(step, token_str) → list   hit + write together
    animate_speed_comparison()    → list   Show speed gain annotation

    Example
    -------
    >>> kv = KVCacheAnim(max_seq_len=10)
    >>> self.play(*kv.animate_init())
    >>> for i, tok in enumerate(["The", "cat", "sat"]):
    ...     self.play(*kv.animate_step(i, tok))
    """

    SLOT_W  = 0.52
    SLOT_H  = 0.80
    SUBH    = 0.32   # K sub-cell height
    SPACING = 0.08

    def __init__(
        self,
        max_seq_len: int = 16,
        d_k: int = 64,
        num_heads: int = 8,
        show_speed_bar: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._max   = max_seq_len
        self._d_k   = d_k
        self._heads = num_heads
        self._filled = 0                    # how many slots are written
        self._slots:   list[VGroup] = []    # all slot VGroups
        self._k_cells: list[Rectangle] = []
        self._v_cells: list[Rectangle] = []

        # --- Title -----------------------------------------------------------
        title_txt = make_text("KV Cache", style=STYLE.COMPONENT_LABEL,
                               color=COMPONENT_COLORS["attention"])
        dim_txt   = make_text(
            f"{num_heads} heads × {d_k} dims",
            style=STYLE.CAPTION,
            color=UI_COLORS["label_dim"],
        )
        dim_txt.next_to(title_txt, RIGHT, buff=0.4)
        header = VGroup(title_txt, dim_txt)

        # --- Slot row --------------------------------------------------------
        slot_row = VGroup()
        for i in range(max_seq_len):
            slot = self._make_slot(empty=True)
            slot_row.add(slot)
            self._slots.append(slot)

        slot_row.arrange(RIGHT, buff=self.SPACING)
        slot_row.next_to(header, DOWN, buff=0.4)

        # Position index labels
        idx_labels = VGroup()
        for i, slot in enumerate(slot_row):
            idx_lbl = make_text(str(i), style=STYLE.MATRIX_VAL,
                                 color=UI_COLORS["label_dim"])
            idx_lbl.next_to(slot, DOWN, buff=0.1)
            idx_labels.add(idx_lbl)

        # Legend
        k_dot = Rectangle(width=0.25, height=0.12,
                          fill_color=COMPONENT_COLORS["key"], fill_opacity=0.8,
                          stroke_width=0)
        v_dot = Rectangle(width=0.25, height=0.12,
                          fill_color=COMPONENT_COLORS["value"], fill_opacity=0.8,
                          stroke_width=0)
        k_lbl = make_text("K", style=STYLE.SMALL, color=COMPONENT_COLORS["key"])
        v_lbl = make_text("V", style=STYLE.SMALL, color=COMPONENT_COLORS["value"])
        k_lbl.next_to(k_dot, RIGHT, buff=0.06)
        v_dot.next_to(k_lbl, RIGHT, buff=0.25)
        v_lbl.next_to(v_dot, RIGHT, buff=0.06)
        legend = VGroup(k_dot, k_lbl, v_dot, v_lbl)
        legend.next_to(slot_row, RIGHT, buff=0.5)

        # Speed bar (optional)
        self._speed_group: VGroup | None = None
        if show_speed_bar:
            self._speed_group = self._build_speed_bar()
            self._speed_group.next_to(slot_row, DOWN, buff=0.9)

        self.add(header, slot_row, idx_labels, legend)
        if self._speed_group is not None:
            self.add(self._speed_group)

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _make_slot(self, empty: bool = True, token: str = "") -> VGroup:
        """Build one KV cache slot (K sub-cell on top, V below)."""
        k_color = UI_COLORS["surface"] if empty else COMPONENT_COLORS["key"]
        v_color = UI_COLORS["surface"] if empty else COMPONENT_COLORS["value"]

        k_cell = Rectangle(
            width=self.SLOT_W, height=self.SUBH,
            fill_color=k_color, fill_opacity=0.6 if not empty else 0.15,
            stroke_color=UI_COLORS["border"], stroke_width=0.8,
        )
        v_cell = Rectangle(
            width=self.SLOT_W, height=self.SUBH,
            fill_color=v_color, fill_opacity=0.6 if not empty else 0.15,
            stroke_color=UI_COLORS["border"], stroke_width=0.8,
        )
        v_cell.next_to(k_cell, DOWN, buff=0.03)

        k_txt = make_text("K" if not empty else "", style=STYLE.TINY if hasattr(STYLE, "TINY") else STYLE.MATRIX_VAL,
                           color=WHITE)
        v_txt = make_text("V" if not empty else "", style=STYLE.MATRIX_VAL, color=WHITE)
        k_txt.move_to(k_cell.get_center())
        v_txt.move_to(v_cell.get_center())

        tok_lbl = make_text(token[:5], style=STYLE.MATRIX_VAL, color=UI_COLORS["label"])
        tok_lbl.next_to(k_cell, UP, buff=0.06)

        return VGroup(k_cell, v_cell, k_txt, v_txt, tok_lbl)

    def _build_speed_bar(self) -> VGroup:
        """Build a 'with vs without cache' speed comparison bar."""
        group = VGroup()
        lbl = make_text("Compute per step", style=STYLE.CAPTION,
                         color=UI_COLORS["label_dim"])

        bar_full = Rectangle(width=3.5, height=0.28,
                             fill_color=SAMPLING_COLORS["temp_hot"],
                             fill_opacity=0.7, stroke_width=0)
        bar_cached = Rectangle(width=0.6, height=0.28,
                               fill_color=SAMPLING_COLORS["top_k"],
                               fill_opacity=0.7, stroke_width=0)

        lbl_full   = make_text("Without cache  (O(n²))", style=STYLE.SMALL,
                                color=SAMPLING_COLORS["temp_hot"])
        lbl_cached = make_text("With cache     (O(n))",  style=STYLE.SMALL,
                                color=SAMPLING_COLORS["top_k"])

        bar_full.next_to(lbl, DOWN, buff=0.2)
        bar_full.to_edge(LEFT, buff=1.0)
        lbl_full.next_to(bar_full, RIGHT, buff=0.15)

        bar_cached.next_to(bar_full, DOWN, buff=0.18)
        bar_cached.align_to(bar_full, LEFT)
        lbl_cached.next_to(bar_cached, RIGHT, buff=0.15)

        group.add(lbl, bar_full, lbl_full, bar_cached, lbl_cached)
        return group

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_init(self) -> list:
        """
        Fade in the empty cache grid.

        Example
        -------
        >>> self.play(*kv.animate_init())
        """
        return [FadeIn(self, run_time=TIMING.NORMAL)]

    def animate_cache_hit(self, up_to_step: int) -> list:
        """
        Flash the already-filled slots (steps 0..up_to_step-1) to show a cache hit.

        Parameters
        ----------
        up_to_step : int
            Number of filled slots to flash.

        Example
        -------
        >>> self.play(*kv.animate_cache_hit(3), run_time=0.4)
        """
        anims = []
        for i in range(min(up_to_step, self._filled)):
            slot = self._slots[i]
            glow = slot.copy().set_stroke(
                UI_COLORS["highlight"], width=3.0
            ).set_fill(opacity=0.9)
            anims.append(
                AnimationGroup(
                    Transform(slot, glow, run_time=TIMING.HIGHLIGHT / 2),
                    Transform(glow, slot, run_time=TIMING.HIGHLIGHT / 2),
                )
            )
        return anims

    def animate_cache_write(self, step: int, token_str: str = "") -> list:
        """
        Fill slot ``step`` with K and V cells and a token label.

        Parameters
        ----------
        step : int
            Slot index to fill (0-based).
        token_str : str
            Token string to display above the slot.

        Example
        -------
        >>> self.play(*kv.animate_cache_write(0, "The"), run_time=0.3)
        """
        if step >= self._max:
            return []

        new_slot = self._make_slot(empty=False, token=token_str)
        # Align to existing slot position
        new_slot.move_to(self._slots[step].get_center())

        self._filled = max(self._filled, step + 1)
        return [
            Transform(self._slots[step], new_slot, run_time=TIMING.KV_CACHE_SHOW),
        ]

    def animate_step(self, step: int, token_str: str = "") -> list:
        """
        Full autoregressive step:  cache hit on past slots + write new slot.

        Example
        -------
        >>> for i, tok in enumerate(["The", "cat", "sat", "on", "the", "mat"]):
        ...     self.play(*kv.animate_step(i, tok))
        """
        return self.animate_cache_hit(step) + self.animate_cache_write(step, token_str)

    def animate_speed_comparison(self) -> list:
        """
        Fade in the speed comparison bar (if ``show_speed_bar=True``).

        Example
        -------
        >>> self.play(*kv.animate_speed_comparison())
        """
        if self._speed_group is None:
            return []
        return [FadeIn(self._speed_group, run_time=TIMING.SLOW)]