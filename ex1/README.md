# 代码说明
> `exN` 中的 `N` = 1

以 `exN.ipynb` 为主，其他文档 / 文件皆为辅助作用；若 `exN.ipynb` 正常运行，则可以无视。*但请勿删除，因其可能被 `exN.ipynb` 引用！*

推荐在 Linux 环境下运行本项目（换行符：LF），其中 python 代码采用 **python 3.6** 运行通过。运行路径最好不要包含中文等 Unicode 字符。如必须在 Windows 等非 Unix 系统中运行，请手动复制项目根目录下的 `assets/` 文件夹至 `exN/` 子目录中，覆盖同名的 `exN/assets`. 由于存在引用，切勿更改任何目录结构。

## 项目结构：
> `tree -I "build|web|__pycache__"`
>
> 请关闭折行（Line-wrapping），否则可能看不清楚～

    exN/
    ├── assets -> ../assets  # Symlink, Windows 下请手动以 ../assets 覆盖
    │
    ├── latex
    │   ├── print.pdf    # 便于打印的文档化 pdf, 但不完整（不含代码及其说明）
    │   ├── print.tex    # TeX 源码，引用 `sections/`
    │   └── sections/    # TeX 分章节源码，其下 `/*.tex` 提供给 `../print.tex`
    │                    # ... 相应地 `/*.pdf` 提供给 `../../exN.ipynb`
    │
    ├── wolfram/         # Mathematica `.nb` 文档，用于生成插图
    │
    ├── ex1.ipynb        # 核心！包罗一切。
    ├── ex1_mini.ipynb   # 备用方案，只包含代码和说明
    ├── ex1_mini.html    # 备用方案，只包含代码和说明（由上一文件生成）
    │
    ├── hilbert_det.py   # \
    ├── hilbert_test.py  #  │-> 独立的 python 源码，节选自 exN.ipynb, 可单独运行
    ├── matrix_sol.py    # /
    │
    └── methodsCompare.csv   # 输出的数据文档
