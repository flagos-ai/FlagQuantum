# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the complete Qualtran v0.7.0 documentation configuration.

"""Render native authored, generated-reference and stored notebook documentation."""

project = "FlagQuantum"
copyright = "2026, FlagQuantum Team"
author = "FlagQuantum Team"

extensions = ["myst_nb"]
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "amsmath", "deflist"]
myst_dmath_double_inline = True
myst_heading_anchors = 3
suppress_warnings = ["myst.header", "myst.nested_header"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "icon_links": [
        {
            "name": "GitHub repository",
            "url": "https://github.com/flagos-ai/FlagQuantum",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        }
    ],
    "footer_center": ["project-logo"],
}
html_sidebars = {
    "concepts/README": ["section-sidebar"],
    "reference/registers/index": ["section-sidebar"],
    "reference/README": ["section-sidebar"],
}
html_static_path = ["_static"]
html_css_files = ["layout.css", "theme.css"]
html_logo = "../assets/logo_flagquantum.png"
html_favicon = "_static/favicon.png"
