LLManim
=======

Reusable Manim components for creating LLM and Transformer explanation videos.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Installation
------------

.. code-block:: bash

   pip install llmanim

Usage
-----

.. code-block:: python

   from manim import *
   from llmanim.base.shapes import MatrixBox, VectorBar, TokenBox

Reference
---------

.. autosummary::
   :toctree: generated
   :recursive:

   llmanim.base.shapes
   llmanim.base.animations
   llmanim.base.utils
   llmanim.attention
   llmanim.blocks
   llmanim.embeddings
   llmanim.feedforward
   llmanim.normalization
   llmanim.output
   llmanim.styles
   llmanim.tokenization
