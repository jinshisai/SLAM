# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


project = 'SLAM'
copyright = '2023, Y.Aso & J.Sai'
author = 'Y.Aso & J.Sai'
project_version = (
    Path(__file__).resolve().parents[1] / 'VERSION'
).read_text().strip()
release = f'v{project_version}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Generate API documentation from docstrings
    'sphinx.ext.napoleon',  # Support Google-style docstrings
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'nbsphinx',  # Support for Jupyter notebook
    'myst_parser',  # Support for Markdown
]

autodoc_member_order = 'bysource'

# -- Source file extentions --------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
