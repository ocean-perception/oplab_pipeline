from correct_images.loaders import default, rosbag, xviii
from oplab import Console


class Loader:
    __slots__ = [
        "bit_depth",
        "_loader",
        "_loader_name",
        "topic",
        "bagfile_list",
        "tz_offset_s",
    ]

    def __init__(self):
        self.bit_depth = None
        self._loader = None
        self._loader_name = None
        self.topic = None
        self.tz_offset_s = 0.0
        self.bagfile_list = []

    def set_loader(self, loader_name):
        if loader_name == "xviii":
            self._loader = xviii.loader
            self._loader_name = "xviii"
        elif loader_name == "default":
            self._loader = default.loader
            self._loader_name = "default"
        elif loader_name == "rosbag":
            self._loader = rosbag.loader
            self._loader_name = "rosbag"
        else:
            Console.quit("The loader type", loader_name, "is not implemented.")
        Console.info("Loader set to", loader_name)

    def set_bagfile_list_and_topic(self, blist, topic):
        self.bagfile_list = blist
        self.topic = topic

    def __call__(self, img_file):
        if self.bit_depth is not None:
            if self._loader_name == "rosbag":
                return self._loader(
                    img_file,
                    self.topic,
                    self.bagfile_list,
                    self.tz_offset_s,
                    self.bit_depth,
                )
            else:
                return self._loader(img_file, src_bit=self.bit_depth)
        else:
            Console.quit("Set the bit_depth in the loader first.")
