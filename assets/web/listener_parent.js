function addListenerParent(pdf_dir, pdf_data) {
    window.addEventListener('message', function(e) {
        if (e.data == 'base64request') {
            framePostData(
                pdf_dir,
                [ pdf_dir, pdf_data ]
            );
        } else {
            frameAction(pdf_dir, function (oneframe) {
                if (e.data[0] == oneframe.name) {
                    oneframe.height = e.data[1] + "px";
                }
            });
        }
    });
}
