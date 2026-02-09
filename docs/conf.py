"""Sphinx configuration file."""

from importlib.metadata import version as get_version

project = "coronagraphoto"
copyright = "2026, Corey Spohn"
author = "Corey Spohn"
release = get_version("coronagraphoto")
version = ".".join(release.split(".")[:2])  # e.g. "1.0" from "1.0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "jupyter_execute"]

language = "Python"

autoapi_dirs = ["../src"]
autoapi_ignore = ["**/*version.py"]
autodoc_typehints = "description"

myst_enable_extensions = ["amsmath", "dollarmath"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
master_doc = "index"
html_title = "coronagraphoto - Coronagraphic Observation Simulator"
html_sidebars = {"posts/*": ["sbt-sidebar-nav.html"]}

html_theme_options = {
    "repository_url": "https://www.github.com/CoreySpohn/coronagraphoto",
    "repository_branch": "main",
    "use_repository_button": True,
    "show_toc_level": 2,
}
html_context = {
    "default_mode": "dark",
}
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}
nb_execution_mode = "auto"  # Execute notebooks without stored outputs
nb_execution_timeout = 120  # Timeout per cell in seconds
nb_execution_raise_on_error = True  # Fail build on notebook errors
