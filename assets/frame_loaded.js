// Thanks: https://stackoverflow.com/a/9163087

function frameDynamicHeight(oneframe) {
    oneframe.height = oneframe.contentWindow.document.body.scrollHeight + "px";
}

function PDFframeLoaded() {
    var allPDFframes = document.getElementsByClassName('PDFframe');
    Array.prototype.forEach.call(allPDFframes, frameDynamicHeight);
}
