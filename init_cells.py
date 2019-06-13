#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa

# <codecell>
# Re-run if you get MathJax gibberish
from assets.specs import startupOption
from assets.pdfshow import pdfshowOption, pdfGet
from assets.startup import initialize
startupOption['debug'] = False
initialize()

# <codecell>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from IPython.display import display, HTML, Markdown, Image

# <codecell>
# If u DON'T HAVE LaTeX, change this to `False`!
# But it would look much better if LaTeX is deployed...
tex_status = True
plt.rc('text', usetex=tex_status)
pager = InlinePager()

# <codecell>
# Basic examples

# # Show docstring:
# show_docstring(Advection1D)
#
# # Pager excerpt:
# pager.extract('Advection1D.pde_solve_upwind', tag='core')
#
# # Timing:
# %%timeit -n 1 -r 1
#
# # Syntax highlighting:
# display(Markdown(
#     """<code class="prettyprint">RuntimeWarning: overflow encountered ...</code>"""
# ))
#
# # Half-way init:
# try:
#     startupOption
# except NameError:
#     %run -i soliton_debug_env.py
