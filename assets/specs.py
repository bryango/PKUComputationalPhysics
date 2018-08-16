#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ != '__main__':  # when IMPORTED
    startupOption = {
        'debug': False,
        'utilities': [
            "matplotlib_notebook",
            "display_whitespace",
            "clear_output",
            "InlinePager",
            "show_docstring",
            "get_methods",
            "goodbye"
        ],
        'reveal': False,
        'initialized': False
    }

    pdfshowOption = {
        'mini': False,
        'notebook_url': ''
    }

if __name__ == '__main__':  # when EXECUTED by assets.startup WITHIN ipynb
    try:
        for func in startupOption['utilities']:
            exec('from assets.startup import ' + func)
        import os  # noqa: F401
        import gc  # noqa: F401
    except NameError:
        raise NameError('startupOption not configured yet!')
