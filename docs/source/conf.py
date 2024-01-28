# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# command line procedures
# 1. sphinx-apidoc -o docs/source ac
# 2. sphinx-build -b html docs/source/ docs/build/html

import os
import sys

sys.path.append(os.path.abspath("../../ac"))

project = "Adiabatic Compiler"
copyright = "2024, Haoyuan Tan & Daniel Huang"
author = "Haoyuan Tan & Daniel Huang"
release = "0.1"

# -- General configuration
extensions = [
    # "sphinx.ext.autodoc",
    # "sphinx.ext.napoleon",
    # "sphinx.ext.autosummary",
    # "sphinx.ext.intersphinx",
    # "nbsphinx",
]

# -- Options for HTML output
html_theme = "sphinx_rtd_theme"


# autodoc_mock_imports = ["tqdm", "numpy"]
# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3/", None),
#     "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
# }
# intersphinx_disabled_domains = ["std"]
# nbsphinx_allow_errors = True
# templates_path = ["_templates"]

# # -- Options for EPUB output
# epub_show_urls = "footnote"
# autosummary_generate = True  # Turn on sphinx.ext.autosummary

# templates_path = ['_templates']
# exclude_patterns = []
