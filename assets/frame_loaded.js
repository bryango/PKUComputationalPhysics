// Thanks: https://stackoverflow.com/a/9163087

function frameDynamicHeight(oneframe) {
    try {
        oneframe.height = oneframe.contentWindow.document.body.scrollHeight + "px";
    } catch(e) {
        console.log("Failed to get height from frame:", oneframe);
    }
}

function PDFframeLoaded() {
    var allPDFframes = document.getElementsByClassName('PDFframe');
    Array.prototype.forEach.call(allPDFframes, frameDynamicHeight);
}
