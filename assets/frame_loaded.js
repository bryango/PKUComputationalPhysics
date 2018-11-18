// Thanks: https://stackoverflow.com/a/9163087

function frameAction(name, func) {
    Array.prototype.forEach.call(
        document.getElementsByName(name),
        func
    );
}

function framePostData(name, data) {
    frameAction(name, function (oneframe) {
        oneframe.contentWindow.postMessage(data, "*");
    });
}

function frameDynamicHeight(oneframe) {
    try {
        oneframe.height = oneframe.contentWindow.document.body.scrollHeight + "px";
    } catch(e) {
        console.log("Failed to get height from frame:", oneframe.name);
    }
}

function PDFframeLoaded() {
    var allPDFframes = document.getElementsByClassName('PDFframe');
    Array.prototype.forEach.call(allPDFframes, frameDynamicHeight);
}
