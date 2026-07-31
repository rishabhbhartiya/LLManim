import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'LLManim'
copyright = '2026, Rishabh Bhartiya'
author = 'Rishabh Bhartiya'
release = '0.1.2'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'myst_parser',
]

autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
html_static_path = ['_static']
