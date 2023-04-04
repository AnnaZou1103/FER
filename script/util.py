import os
import shutil


def make_dir(file_dir):
    if os.path.exists(file_dir):
        print('Directory exists')
        shutil.rmtree(file_dir)
        os.makedirs(file_dir)
    else:
        os.makedirs(file_dir)
