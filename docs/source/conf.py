import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "AthenaKit"
copyright = "2026, AthenaKit Developers"
author = "Minghao Guo"
release = "1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "AthenaKit documentation"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#3d5a73",
        "color-brand-content": "#3d5a73",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8fb0c9",
        "color-brand-content": "#8fb0c9",
    },
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "attrs_inline",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__, __call__, __getitem__",
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_mock_imports = ["pyxsim", "yt", "cupy"]

suppress_warnings = ["ref.python", "ref.ref"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
