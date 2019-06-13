# IPython Notebook 自用模板
基于 [bryango/PKUComputationalPhysics](https://github.com/bryango/PKUComputationalPhysics.git).

**Warnings:**

以 `.ipynb` 为主，其他文档 / 文件皆为辅助作用；若 `.ipynb` 正常运行，则可以无视。*但请勿删除，因其可能被 `.ipynb` 引用！*

推荐在 Linux 环境下运行本项目（换行符：LF），其中 python 代码采用 **Python 3.7.3** 运行通过（很遗憾，Python 3.5 不支持最方便的 f-string, 故至少要求 Python 3.6）。运行路径最好不要包含中文等 Unicode 字符。由于存在引用，切勿更改任何目录结构。

**Features:**

- 核心数学推演由 LaTeX 编写并导出为 PDF, 利用 PDF.js 无损嵌入 `.ipynb`
- 提供额外的代码高亮模块：Google's code prettifier, 以高亮 `.ipynb` 的行内代码
- 另有更多对 `.ipynb` 的个性化设置，整合在启动脚本 `assets/startup.py`

**Notes:**

- 为使用 PDF 嵌入功能，需要 clone PDF.js, 详见 `assets/README.md`;
- 也可直接使用 `--recursive --shallow-submodules` 选项

## 项目结构：
> `tree -d -I "build|web|__pycache__"`
>
> 下面请关闭折行（Line-wrapping），否则可能看不清楚～

    ./
    │
    ├── .ipynb              # 核心！主文档
    │
    ├── assets              # `.ipynb` 配套工具
    │   ├── ...             # ... 实现在 `.ipynb` 中无损显示 PDF, & more!
    │   ├── pdfjs-dist/     # ... 基于 PDF.js 及其示例
    │   └── code-prettify/  # Google code prettify, 额外的代码高亮模块供选
    │
    ├── pycode
    │   ├── backstage.py    # 重要！`.ipynb` 的幕后工作，被其引用
    │   │
    │   ├── toolkit         # 重要！计算物理核心代码，作为 module
    │   │   ├── generic.py  # ... 这还包含一些常用小功能，如求相对误差
    │   │   └── *.py        # ... 还有往往空白的 `__init__.py` 等等
    │   │
    │   └── *.py            # 与 toolkit 的通用性不同，...
    │                       # ... 这里还有些仅适用于此项目的代码
    │                       # ... 以及往往空白的 `__init__.py`
    ├── latex
    │   ├── document.pdf    # LaTeX 文档
    │   ├── document.tex    # TeX 源码，引用 `sections/`
    │   │
    │   │   # 下面为辅助性的 TeX 文档
    │   │
    │   ├── macros.tex      # 自定义 TeX macros;
    │   │                   # ... 提供给其他 `.tex` 文档和 `.ipynb`
    │   ├── mathjax.tex     # 适用于 MathJax 的额外命令;
    │   │                   # ... 仅供 `.ipynb` 使用
    │   ├── packages.tex    # LaTeX 宏包及 preamble 设置
    │   │                   # ... 仅供给其他 `.tex` 文档
    │   │
    │   └── sections        # 分章节文档
    │       ├── `*.tex`     # ... 其下 `/*.tex` 供给 `../document.tex`
    │       ├── `*.pdf`     # ... 相应地 `/*.pdf` 供给 `.ipynb`
    │       │
    │       └── standalone.tex  # 分章节文档所需的宏包及 preamble
    │
    ├── ...   # 可能还有些散落在外的资源（如本 `README.md` 和一些图片）
    └── csv/  # 输出的数据文档及由此生成的资源



**感谢：** [PDF.js](https://github.com/mozilla/pdf.js/), [Google code prettifier](https://github.com/google/code-prettify)
