import numpy as np
import cv2
import os
import h5py
from retinaface import RetinaFace

datapath = os.path.join('data_refined_all.h5')

img_dir = 'KDEF/'
img_test_dir = 'KDEF_and_AKDEF/KDEF_/'

emotion = ['AN', 'DI', 'AF', 'HA', 'SA', 'SU', 'NE']

Training_x = []
Training_y = []
PublicTest_x = []
PublicTest_y = []

count=0
idx=0

for dir in os.listdir(img_test_dir):
    idx+=1
    if idx%100==0:
        print(idx)
    dir = img_test_dir + str(dir)

    for img in os.listdir(dir):
        img_array = cv2.imread(os.path.join(dir, img))

        if np.shape(img_array)!=(762,562,3):
            img_array = cv2.resize(img_array, (562, 762))

        faces = RetinaFace.detect_faces(img_array, threshold=0.5)

        if isinstance(faces, tuple):
            count += 1
            crop_img = img_array

            PublicTest_y.append(emotion.index(img[4:6]))
            PublicTest_x.append(crop_img)
        else:
            for key in faces.keys():
                identity = faces[key]
                facial_area = identity['facial_area']

                if facial_area[3] - facial_area[1] <= 10 or facial_area[2] - facial_area[0] <= 10:
                    crop_img = img_array
                    count += 1
                else:
                    crop_img = img_array[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]
                    crop_img = cv2.resize(crop_img, (562, 762))

                PublicTest_y.append(emotion.index(img[4:6]))
                PublicTest_x.append(crop_img)

print('unrecognized_face: ', count)
print(np.shape(Training_x))
print(np.shape(PublicTest_x))
print(np.shape(PublicTest_y))

datafile = h5py.File(datapath, 'w')
datafile.create_dataset("Training_pixel", dtype = 'uint8', data=Training_x)
datafile.create_dataset("Training_label", dtype = 'int64', data=Training_y)
datafile.create_dataset("PublicTest_pixel", dtype = 'uint8', data=PublicTest_x)
datafile.create_dataset("PublicTest_label", dtype = 'int64', data=PublicTest_y)
datafile.close()