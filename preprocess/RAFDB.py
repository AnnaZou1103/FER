import glob
import os
import shutil

idx=0
file_dir='../dataset/RAFDBAligned'
image_list = glob.glob('RAFDBAligned/train_*.jpg')

label_file = open('list_patition_label.txt', 'r')
labels = label_file.readlines()

if os.path.exists(file_dir):
    print('true')
    shutil.rmtree(file_dir)
    os.makedirs(file_dir)
else:
    os.makedirs(file_dir)

from sklearn.model_selection import train_test_split

trainval_files, val_files = train_test_split(image_list, test_size=0.2, random_state=42)

for img in trainval_files:
    idx+=1
    if idx%100==0:
        print(idx)

    img_number = int(img.split('_')[1].split('.')[0])
    file_class = labels[img_number - 1][-2]
    file_class = os.path.join(file_dir + '/train/', file_class)

    if not os.path.isdir(file_class):
        os.makedirs(file_class)

    shutil.copy(img, file_class + '/' + img.split('/')[1])

for img in val_files:
    idx += 1
    if idx % 100 == 0:
        print(idx)
    img_number = int(img.split('_')[1].split('.')[0])
    file_class = labels[img_number - 1][-2]
    file_class = os.path.join(file_dir + '/val/', file_class)

    if not os.path.isdir(file_class):
        os.makedirs(file_class)

    shutil.copy(img, file_class + '/' + img.split('/')[1])