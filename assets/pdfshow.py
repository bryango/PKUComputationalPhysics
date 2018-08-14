#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from IPython.display import display, HTML
import os.path

pdfshowOption = {
    'mini': False
}


def pdf_autoreload_script():
    return """<script src="assets/frame_loaded.js"></script>\n"""


class pdfGet(object):
    """ Get PDF directory and show in ipynb """

    def __init__(self, pdf_dir):

        if (type(pdf_dir) is str
            and os.path.isfile(pdf_dir)
                and os.path.splitext(pdf_dir)[-1].lower() == '.pdf'):
                    self.pdfDir = pdf_dir
        else:
            self.pdfDir = 'assets/maxwell.pdf'

    def show(self):
        frameJS = f"""
            <p style="font-size: 12px; font-style: italic; ">
                See no PDF below?
                Go to <a href="{self.pdfDir}">{self.pdfDir}</a> directly! </p>
            <iframe class="PDFframe" src='assets/embed.html?file={self.pdfDir}'
                width="100%" frameborder="0" onload="PDFframeLoaded()" >
            </iframe>
        """ if pdfshowOption['mini'] is False else f"""
            <p style="font-size: 12px; font-style: italic; ">
                Mini mode activated!
                Source: <a href="{self.pdfDir}">{self.pdfDir}</a></p>
            <iframe src="{self.pdfDir}#view=fitH"
                width="100%" height="360px" frameborder="0"
                onload="PDFframeLoaded()" >
            </iframe>
        """
        display(HTML(frameJS))
