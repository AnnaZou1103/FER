import glob
import os
import shutil
from sklearn.model_selection import train_test_split


def make_dir(file_dir):
    if os.path.exists(file_dir):
        print('Directory exists')
        shutil.rmtree(file_dir)
        os.makedirs(file_dir)
    else:
        os.makedirs(file_dir)


def split_data(dataset_dir, files, set_name='/train/'):
    idx = 0
    for img in files:
        idx += 1
        if idx % 100 == 0:
            print(idx)

        img_number = int(img.split('/')[-1].split('_')[1].split('.')[0])
        file_class = labels[img_number - 1][-2]
        file_class = os.path.join(dataset_dir + set_name, file_class)

        if not os.path.isdir(file_class):
            os.makedirs(file_class)

        shutil.copy(img, file_class + '/' + img.split('/')[1])


if __name__ == '__main__':
    file_dir = '../dataset/RAFDBAligned'
    image_list = glob.glob('../dataset/original/RAFDBAligned/train_*.jpg')

    label_file = open('../dataset/original/list_patition_label.txt', 'r')
    labels = label_file.readlines()

    make_dir(file_dir)

    train_files, val_files = train_test_split(image_list, test_size=0.2, random_state=42)

    test_files = glob.glob('../dataset/original/RAFDBAligned/test_*.jpg')

    split_data(file_dir, train_files, set_name='/train/')
    split_data(file_dir, val_files, set_name='/val/')
    split_data(file_dir, test_files, set_name='/test/')
