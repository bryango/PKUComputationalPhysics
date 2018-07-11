#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa

# Re-run if you get MathJax gibberish
from assets.specs import startupOption
from assets.pdfshow import pdfshowOption, pdfGet
from assets.startup import initialize
startupOption['debug'] = False
initialize()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Markdown, Image

# LaTeX off for better performance
tex_status = False
plt.rc('text', usetex=tex_status)
plt.rc('text.latex', unicode=tex_status)
pager = InlinePager()
