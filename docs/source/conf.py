# -- Path Setup ----------------------------------------------------------------------

# if extensions (or modules to document with autodoc) are in another directory add these directories
# to sys.path here. If the directory is relative to the documentation root, use
# os.path.abspath to make it absolute
import os
import sys

# sys.path.insert(0, os.path.abspath('../../'))
# print("CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======S")
# Note:
print(os.getcwd())
path = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
frontend_path = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "frontend")
)
interpreter_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "interpreter"))

print("path: ", path)


def list_directories(path):
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories


directories_list = list_directories(path)


print("List of directories in {}: {}".format(path, directories_list))
print("List of directories in {}: {}".format(frontend_path, directories_list))

sys.path.append(path)
sys.path.append(frontend_path)
sys.path.append(interpreter_path)


# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "CompileToAdiabatic"
copyright = "The CompileToAdiabatic authors"
author = "Haoyuan Tan & Daniel Huang"

release = "0.1"
version = "0.1.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx.ext.napoleon",
]
autodoc_mock_imports = ["tqdm", "numpy"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]
nbsphinx_allow_errors = True
templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
autosummary_generate = True  # Turn on sphinx.ext.autosummary
