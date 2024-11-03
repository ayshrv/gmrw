import os

import dominate
from dominate.tags import *

js_script_src = (
    "https://cdn.jsdelivr.net/npm/vanilla-lazyload@17.8.3/dist/lazyload.min.js"
)

js_script = """
        (function () {
            function logElementEvent(eventName, element) {
            console.log(Date.now(), eventName, element.getAttribute('data-src'));
            }

            var callback_enter = function (element) {
            logElementEvent('ENTERED', element);
            };
            var callback_exit = function (element) {
            logElementEvent('EXITED', element);
            };
            var callback_loading = function (element) {
            logElementEvent('LOADING', element);
            };
            var callback_loaded = function (element) {
            logElementEvent('LOADED', element);
            };
            var callback_error = function (element) {
            logElementEvent('ERROR', element);
            element.src = 'https://via.placeholder.com/440x560/?text=Error+Placeholder';
            };
            var callback_finish = function () {
            logElementEvent('FINISHED', document.documentElement);
            };
            var callback_cancel = function (element) {
            logElementEvent('CANCEL', element);
            };

            var ll = new LazyLoad({
            class_applied: 'lz-applied',
            class_loading: 'lz-loading',
            class_loaded: 'lz-loaded',
            class_error: 'lz-error',
            class_entered: 'lz-entered',
            class_exited: 'lz-exited',
            // Assign the callbacks defined above
            callback_enter: callback_enter,
            callback_exit: callback_exit,
            callback_cancel: callback_cancel,
            callback_loading: callback_loading,
            callback_loaded: callback_loaded,
            callback_error: callback_error,
            callback_finish: callback_finish
            });
        })();
    """


class HTML:
    def __init__(self, web_dir, title, html_filename=None, img_dir=None, refresh=0):
        if html_filename is None:
            self.html_filename = "index.html"
        else:
            self.html_filename = html_filename

        self.title = title
        self.web_dir = web_dir
        if img_dir is None:
            self.img_dir = os.path.join(self.web_dir, "images")
            self.img_folder_name = "images"
        else:
            self.img_dir = os.path.join(self.web_dir, img_dir)
            self.img_folder_name = img_dir
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

        self.row_counter = 1

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images_in_one_row(self, ims, txts, links, width=512):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(
                        style="word-wrap: break-word;", halign="center", valign="top"
                    ):
                        with p():
                            with a(href=os.path.join(self.img_folder_name, link)):
                                img(
                                    _class="lazy",
                                    style="width:%dpx" % (width),
                                    data_src=os.path.join(self.img_folder_name, im),
                                )
                            br()
                            p(txt)

    def add_images_in_rows(
        self,
        ims,
        txts,
        links,
        width=512,
        video_height=512,
        add_new_table=True,
        is_video=False,
        lazy_load=True,
    ):
        if add_new_table:
            self.add_table()
        with self.t:
            for im, txt, link in zip(ims, txts, links):
                with tr():
                    with td(
                        style="word-wrap: break-word;", halign="center", valign="top"
                    ):
                        with p():
                            with a(href=os.path.join(self.img_folder_name, link)):
                                if is_video is False:
                                    if lazy_load:
                                        img(
                                            _class="lazy",
                                            style="width:%dpx" % (width),
                                            data_src=os.path.join(
                                                self.img_folder_name, im
                                            ),
                                        )
                                    else:
                                        img(
                                            style="width:%dpx" % (width),
                                            src=os.path.join(self.img_folder_name, im),
                                        )
                                else:
                                    with video(
                                        width=width,
                                        height=video_height,
                                        loop="true",
                                        autoplay="autoplay",
                                    ):
                                        if lazy_load:
                                            source(
                                                _class="lazy",
                                                data_src=os.path.join(
                                                    self.img_folder_name, im
                                                ),
                                                type="video/mp4",
                                            )
                                        else:
                                            source(
                                                src=os.path.join(
                                                    self.img_folder_name, im
                                                ),
                                                type="video/mp4",
                                            )
                            br()
                            txt_with_num = f"{self.row_counter}. {txt}"
                            p(txt_with_num)
                    self.row_counter += 1

    def save(self):
        html_file = "%s/%s" % (self.web_dir, self.html_filename)

        with self.doc.body:
            script(src=js_script_src)
            script(js_script)

        f = open(html_file, "wt")
        f.write(self.doc.render())
        f.close()


if __name__ == "__main__":
    html = HTML("predictions/", "test_html", img_dir="train", refresh=10)
    html.add_header("v8_raft_randInit")

    ims = []
    txts = []
    links = []
    for step in range(1):
        for batch_idx in range(8):
            ims.append(f"step{step}_batch{batch_idx}.mp4")
            txts.append(f"step{step}_batch{batch_idx}.mp4")
            links.append(f"step{step}_batch{batch_idx}.mp4")
    html.add_images_in_rows(
        ims, txts, links, width=2858, video_height=446, is_video=True
    )  # 2858 × 446
    html.save()

    ims = []
    txts = []
    links = []
    for step in range(1, 2):
        for batch_idx in range(8):
            ims.append(f"step{step}_batch{batch_idx}.mp4")
            txts.append(f"step{step}_batch{batch_idx}.mp4")
            links.append(f"step{step}_batch{batch_idx}.mp4")
    html.add_images_in_rows(
        ims,
        txts,
        links,
        width=2858,
        video_height=446,
        is_video=True,
        add_new_table=False,
    )  # 2858 × 446
    # html.add_images_in_one_row(ims, txts, links, width=2048)
    html.save()
