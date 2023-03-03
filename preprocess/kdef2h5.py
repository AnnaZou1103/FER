import numpy as np
import cv2
import os
import h5py

datapath = os.path.join('data_all.h5')

img_dir = 'KDEF_and_AKDEF/KDEF/'
img_test_dir = 'KDEF_and_AKDEF/KDEF_test/'
emotion = ['AN', 'DI', 'AF', 'HA', 'SA', 'SU', 'NE']

Training_x = []
Training_y = []
PublicTest_x = []
PublicTest_y = []

idx=0

for dir in os.listdir(img_dir):
    dir = img_dir + str(dir)

    if dir == 'KDEF_and_AKDEF/KDEF/.DS_Store':
        continue

    for img in os.listdir(dir):
        img_array = cv2.imread(os.path.join(dir, img))
        idx += 1
        if idx % 100==0:
            print(idx)

        if np.shape(img_array)!=(762,562,3):
            img_array = cv2.resize(img_array, (562, 762))

        Training_y.append(emotion.index(img[4:6]))
        Training_x.append(img_array)

for dir in os.listdir(img_test_dir):
    dir = img_test_dir + str(dir)

    if dir == 'KDEF_and_AKDEF/KDEF_test/.DS_Store':
        continue

    for img in os.listdir(dir):
        if img == '.DS_Store':
            continue
        img_array = cv2.imread(os.path.join(dir, img))

        if np.shape(img_array)!=(762,562,3):
            img_array = cv2.resize(img_array, (562, 762))

        PublicTest_y.append(emotion.index(img[4:6]))
        PublicTest_x.append(img_array)

print(np.shape(Training_x))
print(np.shape(PublicTest_x))
print(np.shape(PublicTest_y))

datafile = h5py.File(datapath, 'w')
datafile.create_dataset("Training_pixel", dtype = 'uint8', data=Training_x)
datafile.create_dataset("Training_label", dtype = 'int64', data=Training_y)
datafile.create_dataset("PublicTest_pixel", dtype = 'uint8', data=PublicTest_x)
datafile.create_dataset("PublicTest_label", dtype = 'int64', data=PublicTest_y)
# # datafile.create_dataset("PrivateTest_pixel", dtype = 'uint8', data=PrivateTest_x)
# # datafile.create_dataset("PrivateTest_label", dtype = 'int64', data=PrivateTest_y)
datafile.close()