#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Basic dependencies
from assets.specs import pdfshowOption
from IPython.display import display, HTML
from urllib.parse import urljoin
import os.path
import base64
# For launching server subprocess
import shlex
import subprocess
import requests


def pdf_helper_html():
    listener_parent_jsfile = "assets/web/listener_parent.js"
    frame_loaded_jsfile = "assets/web/frame_loaded.js"
    return "\n".join([
        f"""<script src="{js_file}"></script>"""
        for js_file in [
            listener_parent_jsfile,
            frame_loaded_jsfile
        ]
    ])


def initialize_server_port(port: int):
    server_proc = subprocess.Popen(
        shlex.split(
            f"python3 -u assets/server.py {port}"
            # `-u` necessary for line buffering
        ),
        stdout=subprocess.PIPE,
        universal_newlines=True
    )
    server_port = int(server_proc.stdout.readline().strip())
    if server_port > 0:
        return server_port
    else:
        raise ChildProcessError(
            f'server failed, returned port: {server_port}'
        )


def url_wrap(url: str):
    # do not serve
    if not pdfshowOption['serve']:
        return url

    # file served
    def url_by_port():
        return f"http://127.0.0.1:{pdfshowOption['server_port']}/{url}"

    if pdfshowOption['server_port'] != 0:
        try:
            req = requests.head(
                url_by_port()
            )
            req.raise_for_status()
            return url_by_port()
        except requests.exceptions.ConnectionError:
            pass

    # (re-)initialize server
    pdfshowOption['server_port'] = initialize_server_port(
        pdfshowOption['server_port']
    )

    return url_by_port()


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
addListenerParent("{self.pdfDir}", "{self.pdfData}");
""")  # noqa: E128

        # HTML assets
        p_tag_start = '<p style="font-size: 12px; font-style: italic;">'
        p_tag_end = f"""
Still nothing? <a href="https://github.com/jupyter/notebook/issues/3652">
Blame jupyter! </a></p>"""

        files_hyperlink = f'<a href="{self.fullDir}">{self.pdfDir}</a>'
        iframe_attrs = f'width="100%" frameborder="0" name="{self.pdfDir}"'
        embed_src = url_wrap('assets/embed.html')
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
