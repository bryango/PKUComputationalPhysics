# PKU 计算物理作业 / Computational Physics
> *Actually, not much physics...*<br/>
> `ex1` 至 `ex3` 本质上 fork 自 @[mirage0-0](https://github.com/mirage0-0) 大神，特此致谢！

**Warnings:**

推荐在 Linux 环境下运行本项目（换行符：LF），其中 python 代码采用 python 3.6 运行通过。运行路径最好不要包含中文等 Unicode 字符。如必须在 Windows 等非 Unix 系统中运行，请手动复制项目根目录下的 `assets/` 文件夹至 `exN/` 子目录中，覆盖同名的 `exN/assets`. 由于存在引用，切勿更改任何目录结构。
> `exN` 中的 `N` = 1, 2, ... 为作业的编号。

**Features:**

- 核心数学推演由 LaTeX 编写并导出为 PDF, 利用 PDF.js 无损嵌入 `.ipynb`
- 提供额外的代码高亮模块：Google's code prettifier, 以高亮 `.ipynb` 的行内代码
- 另有更多对 `.ipynb` 的个性化设置，整合在启动脚本 `assets/startup.py`

**Notes:**

- 作业原题版权属于刘川老师与李强老师，不在此贴上（但答案往往足以自明）
- 全功能的 `.ipynb` 文档请见 `ex5`, `ex6`, 本人初学 python, `ex1` 至 `ex3` 建议只读 <img src="https://bryango.github.io/assets/coolemoji/d_erha.png" width="24px"/>

## 项目结构：
> `tree -d -I "build|web|__pycache__"`
>
> 下面请关闭折行（Line-wrapping），否则可能看不清楚～

    ├── assets                # `.ipynb` 配套工具
    │   ├── ...               # ... 实现在 `.ipynb` 中无损显示 PDF, & more!
    │   ├── pdfjs-dist/       # ... 基于 PDF.js 及其示例
    │   └── code-prettify/    # Google code prettify, 额外的代码高亮模块供选
    │
    ├── exN                      # 作业核心部分
    │   └── assets -> ../assets  # Symlink, Windows 下请手动以 ../assets 覆盖
    │   ...

*关于 `exN/` 子目录下文件的更具体说明，请参见 `exN/` 下的 `README.md` 文档。*

**感谢：** [PDF.js](https://github.com/mozilla/pdf.js/), [Google code prettifier](https://github.com/google/code-prettify)
