import os
import shutil


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def make_dir(file_dir):
    if os.path.exists(file_dir):
        print('Directory exists')
        shutil.rmtree(file_dir)
        os.makedirs(file_dir)
    else:
        os.makedirs(file_dir)
