
//// full url
function full_url(relative_url) {
    var link = document.createElement("a");
    link.href = relative_url;
    return (link.protocol+"//"+link.host+link.pathname+link.search+link.hash);
}
var root_url = full_url('/')

//// `files` url
function files_url(relative_url) {
    var relative_to_root = full_url(relative_url).replace(
        root_url, ''
    ).replace(
        'notebooks', 'files'
    );
    return (root_url + relative_to_root);
}

//// Get notebook `/files/` url
var command = ('pdfshowOption["notebook_url"] = notebook_files = "'
    + files_url('./') + '"');
IPython.notebook.kernel.execute(command);

//// Always expand output area (legacy but powerful hack)
require(
    ["notebook/js/outputarea"],
    function (oa) {
        oa.OutputArea.prototype._should_scroll = function(lines) {
            return false;
        }
    }
);

//// Setting auto_scroll_threshold to -1 (latest but not as good)
require(
    ["notebook/js/outputarea"],
    function (oa) {
        oa.OutputArea.auto_scroll_threshold = -1;
    }
);

//// Bonus: inline highlighting with code-prettify
$([IPython.events]).on("rendered.MarkdownCell", function () {
    PR.prettyPrint();
});

//// Auto-save whenever output is generated
$([IPython.events]).on("output_appended.OutputArea", function () {
    IPython.notebook.save_notebook();
});
