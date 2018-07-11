# 代码说明
> `exN` 中的 `N` = 4

以 `exN.ipynb` 为主，其他文档 / 文件皆为辅助作用；若 `exN.ipynb` 正常运行，则可以无视。*但请勿删除，因其可能被 `exN.ipynb` 引用！*

**更新：** 自 `N` = 2, 起，采用 Windows-友好 的打包方式；对于以 `zip` 格式分发的程序，不再需要担心 Symlink 问题。Hoo-ray!

推荐在 Linux 环境下运行本项目（换行符：LF），其中 python 代码采用 **IPython 3.6.4** 运行通过（很遗憾，python 3.5 不支持最方便的 string formatting）。运行路径最好不要包含中文等 Unicode 字符。由于存在引用，切勿更改任何目录结构。

## 项目结构：
> `tree -I "build|web|__pycache__"`
>
> 下面请关闭折行（Line-wrapping），否则可能看不清楚～

    exN/
    │
    ├── exN.ipynb        # 核心！主文档，包罗一切。
    ├── exN_mini.html    # 由上一文件生成，采用兼容性的 PDF 嵌入方案
    │
    ├── assets/ -> ../assets/   # Symlink, 打包时将以源文件覆盖
    │   ├── ...                 # ... 实现在 `.ipynb` 中无损显示 PDF
    │   └── pdfjs-dist/         # ... 基于 PDF.js 及其示例
    │
    ├── pycode
    │   ├── backstage.py    # 重要！`.ipynb` 的幕后工作，被其引用
    │   │
    │   ├── toolkit         # 重要！！计算物理核心代码，作为 module
    │   │   ├── generic.py  # ... 这还包含一些常用小功能，如求相对误差
    │   │   └── *.py        # ... 还有往往空白的 `__init__.py` 等等
    │   │
    │   └── *.py            # 与 toolkit 的通用性不同，...
    │                       # ... 这里还有些仅适用于本次作业的代码
    │                       # ... 以及往往空白的 `__init__.py`
    ├── latex
    │   ├── document.pdf  # 理论，并不完整（不含数值结论、代码及说明）
    │   ├── document.tex  # TeX 源码，引用 `sections/`
    │   │
    │   │   # 下面为辅助性的 TeX 文档
    │   │
    │   ├── macros.tex    # 自定义 TeX macros;
    │   │                 # ... 提供给其他 `.tex` 文档和 `../exN.ipynb`
    │   ├── mathjax.tex   # 适用于 MathJax 的额外命令;
    │   │                 # ... 仅供 `../exN.ipynb` 使用
    │   ├── packages.tex  # LaTeX 宏包及 preamble 设置
    │   │                 # ... 仅供给其他 `.tex` 文档
    │   │
    │   └── sections      # 分章节文档
    │       ├── `*.tex`   # ... 其下 `/*.tex` 供给 `../document.tex`
    │       ├── `*.pdf`   # ... 相应地 `/*.pdf` 供给 `../../exN.ipynb`
    │       │
    │       └── standalone.tex  # 分章节文档所需的宏包及 preamble
    │
    ├── ...   # 可能还有些散落在外的资源（如本 `README.md` 和一些图片）
    └── csv/  # 输出的数据文档（在有作图的情况下并无多大用处）

**感谢：** [PDF.js 项目](https://github.com/mozilla/pdf.js/)
