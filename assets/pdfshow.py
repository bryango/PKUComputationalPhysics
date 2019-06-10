#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from IPython.display import display, HTML
from urllib.parse import urljoin
from assets.specs import pdfshowOption
import os.path
import base64


def pdf_autoreload_html():
    return """<script src="assets/web/frame_loaded.js"></script>"""


class pdfGet(object):
    """ Get PDF directory and show in ipynb """

    def __init__(self, pdf_dir):
        # Try to find a base url
        if pdfshowOption['notebook_url'] == '':
            try:
                # `notebook_files` from `startup.py` javascript
                pdfshowOption['notebook_url'] = notebook_files
            except NameError:
                pass

        if (type(pdf_dir) is str
            and os.path.isfile(pdf_dir)
                and os.path.splitext(pdf_dir)[-1].lower() == '.pdf'):
                    self.pdfDir = pdf_dir
        else:
            self.pdfDir = 'assets/web/maxwell.pdf'

        # Simply read data as base64
        with open(self.pdfDir, 'rb') as pdf_file:
            self.pdfData = base64.b64encode(pdf_file.read()).decode()

        # Use base url directory
        self.fullDir = urljoin(
            pdfshowOption['notebook_url'],
            self.pdfDir
        )

    def show(self):

        # Listen for page dimensions
        get_ipython().run_cell_magic('javascript', '',  # noqa: F821
            f"""
window.addEventListener('message', function(e) {{
    if (e.data == 'based64request') {{
        framePostData(
            "{self.pdfDir}",
            ["{self.pdfDir}", "{self.pdfData}"]
        );
    }} else {{
        frameAction("{self.pdfDir}", function (oneframe) {{
            if (e.data[0] == oneframe.name) {{
                oneframe.height = e.data[1] + "px";
            }}
        }});
    }}
}});""")  # noqa: E128

        # HTML assets
        p_tag_start = '<p style="font-size: 12px; font-style: italic;">'
        p_tag_end = f"""
Still nothing? <a href="https://github.com/jupyter/notebook/issues/3652">
Blame jupyter! </a></p>"""

        files_hyperlink = f'<a href="{self.fullDir}">{self.pdfDir}</a>'
        iframe_attrs = f'width="100%" frameborder="0" name="{self.pdfDir}"'
        embed_src = pdfshowOption['notebook_url'] \
            + 'assets/web/embed.html'
        frameJS = f"""
{p_tag_start}See no PDF below? Go to {files_hyperlink} directly! {p_tag_end}
<iframe class="PDFframe"
    src='{embed_src}?file={self.pdfDir}'
    {iframe_attrs} onload="PDFframeLoaded()" >
</iframe>
""" if pdfshowOption['mini'] is False else f"""
{p_tag_start}Mini mode activated! Source: {files_hyperlink} {p_tag_end}
<iframe src="{self.fullDir}#view=fitH"
    {iframe_attrs} height="360px" >
</iframe>
"""
        display(HTML(frameJS))
