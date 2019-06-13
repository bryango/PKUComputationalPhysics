# IPynb 的格式控制与 PDF 支持

## 依赖 PDF.js
获取最新版 pdfjs-dist, 忽略历史 commits:
```bash
git clone --depth=1 https://github.com/mozilla/pdfjs-dist.git
```
> 参见 <https://stackoverflow.com/a/1210012>.

## 目录结构与说明：
> `tree -I "build|web|__pycache__"`
>
> 下面请关闭折行（Line-wrapping），否则可能看不清楚～

    assets/
    │
    ├── code-prettify/  # Google code prettify, 额外的代码高亮模块
    ├── pdfjs-dist/     # PDF.js pre-built 库，实现在 `.ipynb` 中无损显示 PDF
    │
    ├── embed.html         # 调用 PDF.js 库，实现 PDF 显示
    ├── web
    │   ├── listener_parent.js  # 监听 `embed.html` 发来的请求
    │   ├── frame_loaded.js     # parent 网页中 `embed.html` 所在框架的后期调整
    │   │                       # ... 实现按内容调整框架大小
    │   │                       # ... 上述两文件由 `pdfshow.py` 调用
    │   │
    │   ├── notebook_init.js    # 启动 notebook 后的个性化设置
    │   ├── style.html          # 启动 notebook 后的界面优化
    │   │                       # ... 上述两文件由 `startup.py` 调用
    │   │
    │   └── maxwell.pdf         # 自制 PDF 样本，作为示例和 fallback 提示
    │
    ├── __init__.py    # 空白
    ├── pdfshow.py     # 在 `.ipynb` 中嵌入 `embed.html` 及 `frame_loaded.js`
    ├── server.py      # serve 本地文件，以绕过 jupyter 的安全限制 [危险！]
    ├── startup.py     # `.ipynb` 的启动脚本，实现个性化的格式控制与初始化
    ├── specs.py       # 参数设置，由 `startup.py` 调用，并在主 namespace 中运行
    │
    └── README.md      # 本 `README.md` 文档
