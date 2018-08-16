#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from IPython.display import display, HTML
from urllib.parse import urljoin
from assets.specs import pdfshowOption
import os.path


def pdf_autoreload_script():
    return """<script src="assets/frame_loaded.js"></script>"""


class pdfGet(object):
    """ Get PDF directory and show in ipynb """

    def __init__(self, pdf_dir):
        # Try to find a base url
        if pdfshowOption['notebook_url'] == '':
            try:
                pdfshowOption['notebook_url'] = notebook_files
            except NameError:
                pass

        if (type(pdf_dir) is str
            and os.path.isfile(pdf_dir)
                and os.path.splitext(pdf_dir)[-1].lower() == '.pdf'):
                    self.pdfDir = pdf_dir
        else:
            self.pdfDir = 'assets/maxwell.pdf'

        # Use base url directory
        self.fullDir = urljoin(
            pdfshowOption['notebook_url'],
            self.pdfDir
        )

    def show(self):
        p_tag = '<p style="font-size: 12px; font-style: italic; ">'
        files_hyperlink = f'<a href="{self.fullDir}">{self.pdfDir}</a>'
        iframe_attrs = 'width="100%" frameborder="0"'
        embed_src = pdfshowOption['notebook_url'] + 'assets/embed.html'
        frameJS = f"""
            {p_tag}
                See no PDF below?
                Go to {files_hyperlink} directly! </p>
            <iframe class="PDFframe"
                src='{embed_src}?file={self.pdfDir}'
                {iframe_attrs} onload="PDFframeLoaded()" >
            </iframe>
        """ if pdfshowOption['mini'] is False else f"""
            {p_tag}
                Mini mode activated!
                Source: {files_hyperlink}</p>
            <iframe src="{self.fullDir}#view=fitH"
                {iframe_attrs} height="360px" >
            </iframe>
        """
        display(HTML(frameJS))
