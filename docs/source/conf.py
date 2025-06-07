# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(1500)



# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'numpydoc',
    'nbsphinx',
    'myst_parser',
    'sphinx.ext.mathjax',  # for math equations
    'sphinxcontrib.bibtex',  # for bibliographic references
    'sphinxcontrib.rsvgconverter',  # for SVG->PDF conversion in LaTeX output
]

nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) | string() %}
.. raw:: html

    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/rs-station/reciprocalspaceship/blob/main/{{ docname|e }}">{{ docname|e }}</a>.
      Interactive online version:
      <span style="white-space: nowrap;">
        <a href="https://mybinder.org/v2/gh/rs-station/reciprocalspaceship/main?filepath={{ docname|e }}?urlpath=lab">
        <img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>.
      </span>
    </div>

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""
nbsphinx_allow_errors = True

# Default language for syntax highlighting in reST and Markdown cells:
highlight_language = 'none'

# Don't add .txt suffix to source files:
html_sourcelink_suffix = ''

# Work-around until https://github.com/sphinx-doc/sphinx/issues/4229 is solved:
html_scaled_image_link = False

# List of arguments to be passed to the kernel that executes the notebooks:
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# This is processed by Jinja2 and inserted after each notebook
nbsphinx_epilog = r"""
{% set docname = 'doc/' + env.doc2path(env.docname, base=None) | string() %}
.. raw:: latex

    \nbsphinxstopnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{\dotfill\ \sphinxcode{\sphinxupquote{\strut
    {{ docname | escape_latex }}}} ends here.}}
"""

mathjax_config = {
    'TeX': {'equationNumbers': {'autoNumber': 'AMS', 'useLabelIds': True}},
}

# Additional files needed for generating LaTeX/PDF output:
# latex_additional_files = ['reference.bib']
bibtex_bibfiles = ['reference.bib']





# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

master_doc = 'index'

project = 'scVAEIT'
author = 'Jin-Hong Du'
copyright = '2025, ' + author

linkcheck_ignore = [r'http://localhost:\d+/']

# -- Get version information and date from Git ----------------------------

try:
    from scVAEIT import __version__
    release = __version__
    # from subprocess import check_output
    # release = check_output(['git', 'describe', '--tags', '--always'])
    # release = release.decode().strip()
    # today = check_output(['git', 'show', '-s', '--format=%ad', '--date=short'])
    # today = today.decode().strip()
except Exception:
    release = '<unknown>'
    # today = '<unknown date>'
# release = '<unknown>'
# -- Options for HTML output ----------------------------------------------

html_title = project

# -- Options for LaTeX output ---------------------------------------------

# # See https://www.sphinx-doc.org/en/master/latex.html
# latex_elements = {
#     'papersize': 'a4paper',
#     'printindex': '',
#     'sphinxsetup': r"""
#         %verbatimwithframe=false,
#         %verbatimwrapslines=false,
#         %verbatimhintsturnover=false,
#         VerbatimColor={HTML}{F5F5F5},
#         VerbatimBorderColor={HTML}{E0E0E0},
#         noteBorderColor={HTML}{E0E0E0},
#         noteborder=1.5pt,
#         warningBorderColor={HTML}{E0E0E0},
#         warningborder=1.5pt,
#         warningBgColor={HTML}{FBFBFB},
#     """,
#     'preamble': r"""
# \usepackage[sc,osf]{mathpazo}
# \linespread{1.05}  % see http://www.tug.dk/FontCatalogue/urwpalladio/
# \renewcommand{\sfdefault}{pplj}  % Palatino instead of sans serif
# \IfFileExists{zlmtt.sty}{
#     \usepackage[light,scaled=1.05]{zlmtt}  % light typewriter font from lmodern
# }{
#     \renewcommand{\ttdefault}{lmtt}  % typewriter font from lmodern
# }
# \usepackage{booktabs}  % for Pandas dataframes
# """,
# }

# latex_documents = [
#     (master_doc, 'nbsphinx.tex', project, author, 'howto'),
# ]

# latex_show_urls = 'footnote'
# latex_show_pagerefs = True

# -- Options for EPUB output ----------------------------------------------

# These are just defined to avoid Sphinx warnings related to EPUB:
version = release
suppress_warnings = ['epub.unknown_project_files']

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints', '**.h5ad']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']




