"""
manim_transformer/styles/fonts.py
===================================
Typography system for all Transformer/LLM animation assets.

Covers:
  - Font family constants (display, body, mono, math)
  - Pre-built Manim Text / MathTex factory functions
  - Consistent text style presets (title, heading, label, caption …)
  - Weight / style helpers
  - A standalone reference runner

Fonts used
----------
All fonts below are free and available via Google Fonts or bundled with
most LaTeX distributions.  Install them on the system before rendering.

  JetBrains Mono       — monospace  / code / token IDs / matrix values
  IBM Plex Sans        — clean sans / UI labels / captions
  IBM Plex Mono        — secondary mono (fallback)
  Outfit               — display headings (geometric, modern)
  Crimson Pro          — serif accent (equation narratives, section titles)
  Fira Math (LaTeX)    — math via MathTex; bundled with MiKTeX/TeX Live

Install (Linux / macOS with pip-installed manim):
    # Download .ttf files from Google Fonts, then:
    mkdir -p ~/.local/share/fonts
    cp *.ttf ~/.local/share/fonts/
    fc-cache -fv

Usage
-----
from manim_transformer.styles.fonts import (
    FONTS,
    STYLE,
    title, heading, subheading,
    body, caption, label, small,
    code, token_word, token_id,
    matrix_val, dimension_label,
    formula, formula_sm,
    make_text, make_code, make_math,
    stacked_label, component_badge,
)
"""

from __future__ import annotations

from manim import Text, MathTex, Tex, VGroup
from manim.utils.color import ManimColor

from llmanim.styles.colors import UI_COLORS
from llmanim.styles.constants import TYPOGRAPHY as T


# ---------------------------------------------------------------------------
# FONT FAMILY NAMES
# ---------------------------------------------------------------------------

class FONTS:
    """
    Central font-family registry.

    All Manim ``Text()`` objects accept a ``font=`` kwarg.
    Use these constants instead of bare strings.
    """

    # -- Primary display / heading -------------------------------------------
    DISPLAY         = "Outfit"
    """
    Geometric sans-serif. Used for scene titles, section headings, big
    callout numbers.  Clean, modern, immediately readable at large sizes.
    """

    # -- Body / UI labels ----------------------------------------------------
    SANS            = "IBM Plex Sans"
    """
    Neutral humanist sans. Body text, UI labels, captions, tooltips.
    Highly legible at small sizes on dark backgrounds.
    """

    # -- Monospace / code / token IDs ----------------------------------------
    MONO            = "JetBrains Mono"
    """
    Developer-grade monospace. Token text, token IDs, matrix cell values,
    any code-like content.  Ligature support; great on dark backgrounds.
    """

    MONO_ALT        = "IBM Plex Mono"
    """
    Secondary monospace fallback (same family as IBM Plex Sans).
    Use when JetBrains Mono is unavailable.
    """

    # -- Serif accent --------------------------------------------------------
    SERIF           = "Crimson Pro"
    """
    Elegant serif. Section narrative text, equation explanations,
    pull-quotes.  Adds visual contrast against the sans UI.
    """

    # -- Math (LaTeX) --------------------------------------------------------
    MATH_TEX        = "Fira Math"
    """
    Used implicitly by MathTex / Tex via LaTeX; not passed as ``font=``.
    Requires the ``fira-math`` LaTeX package.
    Fallback: default Computer Modern if not installed.
    """

    # -- Fallbacks (always available, no install needed) ---------------------
    FALLBACK_SANS   = "DejaVu Sans"
    FALLBACK_MONO   = "DejaVu Sans Mono"
    FALLBACK_SERIF  = "DejaVu Serif"

    @classmethod
    def all(cls) -> dict[str, str]:
        """Return all public font names as ``{attr: family}`` dict."""
        return {
            k: v for k, v in vars(cls).items()
            if not k.startswith("_") and isinstance(v, str)
        }


# ---------------------------------------------------------------------------
# TEXT STYLE PRESETS
# ---------------------------------------------------------------------------
# Each preset is a dict of kwargs forwarded to ``Text()`` or ``make_text()``.
# Usage:  Text("Hello", **STYLE.TITLE)

class STYLE:
    """Pre-built keyword-argument bundles for ``Text()``."""

    TITLE = dict(
        font        = FONTS.DISPLAY,
        font_size   = T.TITLE,
        color       = UI_COLORS["label"],
        weight      = "BOLD",
    )

    HEADING = dict(
        font        = FONTS.DISPLAY,
        font_size   = T.HEADING,
        color       = UI_COLORS["label"],
        weight      = "BOLD",
    )

    SUBHEADING = dict(
        font        = FONTS.SANS,
        font_size   = T.SUBHEADING,
        color       = UI_COLORS["label"],
        weight      = "MEDIUM",
    )

    BODY = dict(
        font        = FONTS.SANS,
        font_size   = T.BODY,
        color       = UI_COLORS["label"],
    )

    CAPTION = dict(
        font        = FONTS.SANS,
        font_size   = T.CAPTION,
        color       = UI_COLORS["label_dim"],
    )

    LABEL = dict(
        font        = FONTS.SANS,
        font_size   = T.BODY,
        color       = UI_COLORS["label"],
    )

    LABEL_DIM = dict(
        font        = FONTS.SANS,
        font_size   = T.CAPTION,
        color       = UI_COLORS["label_dim"],
    )

    SMALL = dict(
        font        = FONTS.SANS,
        font_size   = T.SMALL,
        color       = UI_COLORS["label_dim"],
    )

    # -- Monospace / code ----------------------------------------------------
    CODE = dict(
        font        = FONTS.MONO,
        font_size   = T.BODY,
        color       = UI_COLORS["label"],
    )

    # -- Token box text ------------------------------------------------------
    TOKEN_WORD = dict(
        font        = FONTS.MONO,
        font_size   = T.TOKEN_WORD,
        color       = UI_COLORS["label"],
        weight      = "BOLD",
    )

    TOKEN_ID = dict(
        font        = FONTS.MONO,
        font_size   = T.TOKEN_ID,
        color       = UI_COLORS["label_dim"],
    )

    # -- Matrix / vector cell values -----------------------------------------
    MATRIX_VAL = dict(
        font        = FONTS.MONO,
        font_size   = T.TINY,
        color       = UI_COLORS["label"],
    )

    # -- Dimension brace labels (e.g. "512 × 768") ---------------------------
    DIMENSION_LABEL = dict(
        font        = FONTS.SANS,
        font_size   = T.SMALL,
        color       = UI_COLORS["highlight"],
        slant       = "ITALIC",
    )

    # -- Math formula as plain text (e.g. "Q = X · W_Q") --------------------
    FORMULA_PLAIN = dict(
        font        = FONTS.SANS,
        font_size   = T.FORMULA,
        color       = UI_COLORS["label"],
        slant       = "ITALIC",
    )

    # -- Component name banner (e.g. "Multi-Head Attention") -----------------
    COMPONENT_LABEL = dict(
        font        = FONTS.DISPLAY,
        font_size   = T.SUBHEADING,
        color       = UI_COLORS["label"],
        weight      = "BOLD",
    )

    # -- Annotation / tooltip / aside ----------------------------------------
    ANNOTATION = dict(
        font        = FONTS.SERIF,
        font_size   = T.CAPTION,
        color       = UI_COLORS["label_dim"],
        slant       = "ITALIC",
    )

    # -- "Future token blocked" / error / warning text -----------------------
    WARNING = dict(
        font        = FONTS.SANS,
        font_size   = T.CAPTION,
        color       = ManimColor("#F44336"),   # UI red
        weight      = "BOLD",
    )

    # -- Positive callout (e.g. "Token attends here") ------------------------
    CALLOUT = dict(
        font        = FONTS.SANS,
        font_size   = T.CAPTION,
        color       = ManimColor("#FFEB3B"),   # highlight yellow
        weight      = "BOLD",
    )


# ---------------------------------------------------------------------------
# FACTORY FUNCTIONS
# ---------------------------------------------------------------------------

def make_text(
    text: str,
    style: dict | None = None,
    color: ManimColor | str | None = None,
    font_size: int | None = None,
    font: str | None = None,
    **kwargs,
) -> Text:
    """
    Create a ``manim.Text`` object from a style preset with optional overrides.

    Parameters
    ----------
    text : str
        The string to display.
    style : dict, optional
        One of the ``STYLE.*`` dicts.  Defaults to ``STYLE.BODY``.
    color : ManimColor | str, optional
        Override the style's color.
    font_size : int, optional
        Override the style's font_size.
    font : str, optional
        Override the style's font family.
    **kwargs
        Any additional kwargs forwarded to ``Text()``.

    Examples
    --------
    >>> make_text("Attention", style=STYLE.HEADING)
    >>> make_text("token_42", style=STYLE.TOKEN_WORD, color="#FFEB3B")
    """
    props = dict(style or STYLE.BODY)   # copy — never mutate the preset
    if color     is not None: props["color"]     = color
    if font_size is not None: props["font_size"] = font_size
    if font      is not None: props["font"]      = font
    props.update(kwargs)
    return Text(text, **props)


def make_code(text: str, color: ManimColor | str | None = None, **kwargs) -> Text:
    """Monospaced code / token text (``STYLE.CODE``)."""
    return make_text(text, style=STYLE.CODE, color=color, **kwargs)


def make_math(
    latex: str,
    font_size: int = T.FORMULA,
    color: ManimColor | str | None = None,
    **kwargs,
) -> MathTex:
    """
    Create a ``manim.MathTex`` object with consistent sizing.

    Parameters
    ----------
    latex : str
        LaTeX math string (no ``$$`` delimiters needed).
    font_size : int
        Font size; defaults to ``TYPOGRAPHY.FORMULA``.
    color : optional
        Text color override.

    Examples
    --------
    >>> make_math(r"\\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V")
    >>> make_math(r"W_Q \\in \\mathbb{R}^{d_{model} \\times d_k}", font_size=18)
    """
    kwargs.setdefault("font_size", font_size)
    if color is not None:
        kwargs["color"] = color
    return MathTex(latex, **kwargs)


def make_tex(latex: str, font_size: int = T.BODY, **kwargs) -> Tex:
    """Create a ``manim.Tex`` object (text-mode LaTeX)."""
    kwargs.setdefault("font_size", font_size)
    return Tex(latex, **kwargs)


# ---------------------------------------------------------------------------
# CONVENIENCE SHORTHANDS
# ---------------------------------------------------------------------------
# One function per semantic role — mirrors CSS utility class naming.

def title(text: str, **kw)           -> Text:     return make_text(text, STYLE.TITLE, **kw)
def heading(text: str, **kw)         -> Text:     return make_text(text, STYLE.HEADING, **kw)
def subheading(text: str, **kw)      -> Text:     return make_text(text, STYLE.SUBHEADING, **kw)
def body(text: str, **kw)            -> Text:     return make_text(text, STYLE.BODY, **kw)
def caption(text: str, **kw)         -> Text:     return make_text(text, STYLE.CAPTION, **kw)
def label(text: str, **kw)           -> Text:     return make_text(text, STYLE.LABEL, **kw)
def label_dim(text: str, **kw)       -> Text:     return make_text(text, STYLE.LABEL_DIM, **kw)
def small(text: str, **kw)           -> Text:     return make_text(text, STYLE.SMALL, **kw)
def code(text: str, **kw)            -> Text:     return make_text(text, STYLE.CODE, **kw)
def token_word(text: str, **kw)      -> Text:     return make_text(text, STYLE.TOKEN_WORD, **kw)
def token_id(text: str, **kw)        -> Text:     return make_text(text, STYLE.TOKEN_ID, **kw)
def matrix_val(text: str, **kw)      -> Text:     return make_text(text, STYLE.MATRIX_VAL, **kw)
def dimension_label(text: str, **kw) -> Text:     return make_text(text, STYLE.DIMENSION_LABEL, **kw)
def formula(latex: str, **kw)        -> MathTex:  return make_math(latex, T.FORMULA, **kw)
def formula_sm(latex: str, **kw)     -> MathTex:  return make_math(latex, T.FORMULA_SM, **kw)
def annotation(text: str, **kw)      -> Text:     return make_text(text, STYLE.ANNOTATION, **kw)
def component_label(text: str, **kw) -> Text:     return make_text(text, STYLE.COMPONENT_LABEL, **kw)
def warning(text: str, **kw)         -> Text:     return make_text(text, STYLE.WARNING, **kw)
def callout(text: str, **kw)         -> Text:     return make_text(text, STYLE.CALLOUT, **kw)


# ---------------------------------------------------------------------------
# MULTI-LINE STACKED LABEL
# ---------------------------------------------------------------------------

def stacked_label(
    lines: list[str],
    styles: list[dict] | None = None,
    spacing: float = 0.1,
    aligned_edge=None,
) -> VGroup:
    """
    Build a vertically stacked group of ``Text`` objects.

    Parameters
    ----------
    lines : list[str]
        Text for each line, top to bottom.
    styles : list[dict], optional
        One ``STYLE.*`` dict per line.  Defaults to ``STYLE.BODY`` for all.
    spacing : float
        Extra vertical gap between lines (Manim units).
    aligned_edge : optional
        Manim direction constant (``LEFT``, ``RIGHT``, ``ORIGIN``).
        Defaults to ``LEFT``.

    Example
    -------
    >>> stacked_label(
    ...     ["Multi-Head Attention", "8 heads × 64 dims"],
    ...     styles=[STYLE.HEADING, STYLE.CAPTION],
    ... )
    """
    from manim import DOWN, LEFT

    aligned_edge = aligned_edge if aligned_edge is not None else LEFT
    styles = styles or [STYLE.BODY] * len(lines)

    group = VGroup()
    for i, (text, style) in enumerate(zip(lines, styles)):
        obj = make_text(text, style)
        if i == 0:
            group.add(obj)
        else:
            obj.next_to(group[-1], DOWN, buff=spacing, aligned_edge=aligned_edge)
            group.add(obj)
    return group


# ---------------------------------------------------------------------------
# COMPONENT BADGE  (colored dot + label)
# ---------------------------------------------------------------------------

def component_badge(name: str, color: ManimColor | str) -> VGroup:
    """
    A colored dot + label pair used to annotate Transformer components.

    Returns a ``VGroup([dot, text])`` laid out left-to-right.

    Parameters
    ----------
    name : str
        Component name, e.g. ``"Attention"``.
    color : ManimColor | str
        The component's canonical color from ``COMPONENT_COLORS``.

    Example
    -------
    >>> from manim_transformer.styles.colors import COMPONENT_COLORS
    >>> badge = component_badge("FFN", COMPONENT_COLORS["ffn"])
    >>> self.add(badge)
    """
    from manim import Dot, RIGHT

    dot = Dot(radius=0.08, color=color)
    txt = make_text(name, style=STYLE.LABEL, color=color)
    txt.next_to(dot, RIGHT, buff=0.12)
    return VGroup(dot, txt)


# ---------------------------------------------------------------------------
# SUBSCRIPT / SUPERSCRIPT HELPER  (plain Text, no LaTeX)
# ---------------------------------------------------------------------------

def sub_label(main: str, sub: str, main_style: dict | None = None) -> VGroup:
    """
    Create a ``main`` text + smaller subscript ``sub`` text pair.

    Useful for labels like "W" + "Q" without spinning up MathTex.

    Example
    -------
    >>> sub_label("W", "Q")     # looks like W_Q
    >>> sub_label("d", "model") # looks like d_model
    """
    from manim import DOWN, RIGHT

    main_obj = make_text(main, style=main_style or STYLE.LABEL)
    sub_obj  = make_text(sub,  style=STYLE.TOKEN_ID)
    sub_obj.next_to(main_obj, RIGHT + DOWN * 0.15, buff=0.03)
    return VGroup(main_obj, sub_obj)


# ---------------------------------------------------------------------------
# QUICK REFERENCE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  manim_transformer  —  Font Reference")
    print("=" * 60)

    print("\n  Font Families  (FONTS.*)")
    print("  " + "-" * 42)
    for attr, family in FONTS.all().items():
        print(f"    FONTS.{attr:<22} {family}")

    print("\n  Style Presets  (STYLE.*)")
    print("  " + "-" * 42)
    for name in dir(STYLE):
        if name.startswith("_"):
            continue
        val = getattr(STYLE, name)
        if isinstance(val, dict):
            print(f"\n    STYLE.{name}")
            for k, v in val.items():
                print(f"      {k:<14} {v}")

    print("\n  Convenience functions")
    print("  " + "-" * 42)
    fns = [
        "title(text)",
        "heading(text)",
        "subheading(text)",
        "body(text)",
        "caption(text)",
        "label(text)",
        "label_dim(text)",
        "small(text)",
        "code(text)",
        "token_word(text)",
        "token_id(text)",
        "matrix_val(text)",
        "dimension_label(text)",
        "formula(latex)",
        "formula_sm(latex)",
        "annotation(text)",
        "component_label(text)",
        "warning(text)",
        "callout(text)",
        "stacked_label(lines, styles)",
        "component_badge(name, color)",
        "sub_label(main, sub)",
    ]
    for fn in fns:
        print(f"    {fn}")