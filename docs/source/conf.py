import os
import sys

# Add the 'src' folder to the Python path
sys.path.insert(0, os.path.abspath("../../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "scDataset"
copyright = "2025, Davide D'Ascenzo"
author = "Davide D'Ascenzo"
version = "0.3.0"
release = "0.3.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_book_theme",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.doctest",  # For testing code blocks in documentation
    "sphinx_sitemap",
    "nbsphinx",  # For including Jupyter notebooks in documentation
]

# nbsphinx settings
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True  # Allow notebooks with errors to be included

# Generate autosummary pages
autosummary_generate = True

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "__weakref__",
}

# Suppress specific warnings
suppress_warnings = ["autosummary.import_cycle"]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "repository_url": "https://github.com/scDataset/scDataset",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs/source",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "",
        "colab_url": "https://colab.research.google.com",
    },
    # Version switcher
    "switcher": {
        "json_url": "https://scdataset.github.io/versions.json",
        "version_match": os.environ.get("DOCS_VERSION", "latest"),
    },
    "navbar_end": ["version-switcher", "navbar-icon-links"],
}

html_title = "scDataset Documentation"
html_logo = "https://github.com/scDataset/scDataset/raw/main/figures/scdataset.png"
html_favicon = (
    "https://github.com/scDataset/scDataset/raw/main/figures/scdataset_favicon.png"
)

# Sitemap configuration
html_baseurl = "https://scdataset.github.io/"
sitemap_url_scheme = "{link}"
