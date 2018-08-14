#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from IPython.core import page
from IPython.display import display, HTML, Markdown, clear_output
from assets.specs import startupOption
from assets.pdfshow import pdf_autoreload_script, pdfshowOption
import os
import re
import sys
import platform
import time
import inspect


class InlinePager(object):
    """ Inline pager """

    def __init__(self):
        self.truncation = (None, None)
        self.page_origin = page.page  # noqa: F841
        self.log = {'output': ''}
        self.truncate()

    def truncate(self, truncation=(None, None), raw=False, full=False):
        self.truncation = truncation

        def page_inline(meta, truncation=truncation, log=self.log):
            if truncation != (None, None):
                if (len(truncation) != 2) or (not all(
                    isinstance(bound, (int, type(None)))
                    for bound in truncation
                )):
                    raise ValueError('page_inline: invalid truncation!')
            clean_output = "\n".join([
                "\x1b[0;31mSource:\x1b[0m",
                *meta["text/plain"]
                .split("\x1b[0;31mSource:\x1b[0m", 1)[1]
                .split("\n")[1:-1][truncation[0]:truncation[1]]
            ]) if not full else meta["text/plain"]
            log['output'] = clean_output
            if not raw:
                print(clean_output)

        page.page = page_inline

    def trim(self, stuff_to_query, truncation=(None, None)):
        """ Returns trimmed pager.
            Deprecated; use `extract` instead.
        """
        self.truncate(truncation)
        info = ("```\n*** Excerpt, "
                f"showing lines[{truncation[0]}:{truncation[1]}] ```")
        display(Markdown(info))
        get_ipython().run_line_magic('pinfo2', stuff_to_query)   # noqa: F821
        display(Markdown(info))
        self.truncate()

    def extract(self, stuff_to_query, tag=''):
        truncation_log = self.truncation
        self.truncate(raw=True)
        info = ("```\n*** Excerpt, "
                f"tag: <{tag}> ```")

        display(Markdown(info))
        get_ipython().run_line_magic('pinfo2', stuff_to_query)   # noqa: F821
        print(re.search(
            f'(^.*# <{tag}>.*$)'
            fr'([\s\S]+?(?=(# </{tag}>)))'
            f'(# </{tag}>)',
            self.log['output'],
            flags=re.MULTILINE
        ).group(0))
        display(Markdown(info))

        self.truncate(truncation_log)


def show_docstring(stuff):
    display_whitespace(0.)
    display(Markdown(f"***Docstring: ***"))
    display(Markdown('```python\n    '
                     + '"""' + stuff.__doc__ + '"""\n'
                     + '```'))


def get_methods(stuff):
    return next(zip(
        *inspect.getmembers(stuff, inspect.isfunction)
    ))


def display_whitespace(margin: float):
    """ Add whitespace at will! """
    display(HTML(f'''<p style="margin-top: {margin}ex"></p>'''))


def matplotlib_notebook():
    """ Interactive plot: run '%matplotlib notebook' three times """
    # Why repeat three times?
    # This is a bug of ipynb:
    # ... see: <stackoverflow.com/a/41251483>
    # ... and: <github.com/jupyter/notebook/issues/473>
    for _ in range(2):
        get_ipython().run_line_magic('matplotlib', 'notebook')   # noqa: F821


def to_flake(filename):
    get_ipython().run_cell_magic('bash', '',         # noqa: F821
        f'filename={filename}\n'
        'mkdir -p tmp\n'
        'jupyter nbconvert $filename.ipynb --stdout --to script'
        ' > tmp/"$filename"_toflake.py')             # noqa: E128


def output_html(filename):
    get_ipython().run_cell_magic('bash', '',         # noqa: F821
        f'filename={filename}\n'
        'jupyter nbconvert $filename.ipynb --stdout --to html'
        ' > "$filename"_mini.html')   # noqa: E128


def save_it():
    get_ipython().run_cell_magic('javascript', '',   # noqa: F821
        'IPython.notebook.save_notebook()')          # noqa: E128
    time.sleep(2)


def goodbye(filename):
    save_it()
    if pdfshowOption['mini'] is True:
        output_html(filename)
    if platform.system() != 'Windows':
        to_flake(filename)
        os.system("printf '\a'")


def initialize():
    # Import in parent namespace
    get_ipython().run_line_magic('run', '-i assets/specs.py')    # noqa: F821

    # Path for pycode/toolkit
    sys.path.append('pycode')
    # sys.path.append('pycode/toolkit')

    # Debug mode
    if startupOption['debug'] is True:
        get_ipython().run_line_magic('load_ext', 'autoreload')   # noqa: F821
        get_ipython().run_line_magic('autoreload', '2')          # noqa: F821

    # Clean up
    clear_output(wait=True)

    # MathJax macros
    with open('latex/macros.tex', 'r') as tex_macros:
        mathjax_macros = tex_macros.read()
    with open('latex/mathjax.tex', 'r') as tex_macros:
        mathjax_macros += tex_macros.read()

    # Inline image & more styles
    # Extra identation to trick Markdown into rendering it as a whole paragraph
    html_style = """
        <link rel="stylesheet" type="text/css"
            href="assets/code-prettify/codestyle.css">
        <script type="text/javascript"
            src="assets/code-prettify/google-code-prettify/run_prettify.js">
        </script>
        <style>
            img.inline_img {
                display: unset;
                margin-top: unset;
            }
            .MathJax_Display {
                margin: 6px;
            }
            .dataframe td {
                white-space: nowrap;
            }
        </style>
        """

    get_ipython().run_cell_magic(  # noqa: F821
        'javascript', '',
        """require(  //// Always expand output area (legacy but powerful hack)
    ["notebook/js/outputarea"],
    function (oa) {
        oa.OutputArea.prototype._should_scroll = function(lines) {
            return false;
        }
    }
);
require(  //// Setting auto_scroll_threshold to -1 (latest but not as good)
    ["notebook/js/outputarea"],
    function (oa) {
        oa.OutputArea.auto_scroll_threshold = -1;
    }
);
$([IPython.events]).on("rendered.MarkdownCell", function () {
    PR.prettyPrint();
});  //// Bonus: inline highlighting with code-prettify
$([IPython.events]).on("output_appended.OutputArea", function () {
    IPython.notebook.save_notebook();
});  //// Auto-save whenever output is generated
""")

    markdown_string = f'${mathjax_macros}$' \
        + html_style \
        + pdf_autoreload_script().rstrip() \
        + """<!--
     --> `< Utilities initialized >`\n""" \
        + "```python\n" \
        + ", ".join(startupOption['utilities']) \
        + "```"

    display(Markdown(markdown_string))

    if startupOption['reveal']:
        print(markdown_string)

    startupOption['initialized'] = True
