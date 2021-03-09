from correct_images.loaders import default, xviii
from oplab import Console

class Loader:
    __slots__ = [
        'bit_depth',
        '_loader'
    ]

    def __init__(self):
        self.bit_depth = None
        self._loader = None

    def set_loader(self, loader_name):
        if loader_name == 'xviii':
            self._loader = xviii.loader
        elif loader_name == 'default':
            self._loader = default.loader
        else:
            Console.quit("The loader type", loader_name, "is not implemented.")
        Console.info('Loader set to', loader_name)

    def __call__(self, img_file):
        if self.bit_depth is not None:
            return self._loader(img_file, src_bit=self.bit_depth)
        else:
            Console.quit("Set the bit_depth in the loader first.")

    