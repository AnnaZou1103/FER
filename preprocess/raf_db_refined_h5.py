import numpy as np
import cv2
import os
import h5py
from retinaface import RetinaFace

datapath = 'raf_db_refined.h5'
label_file = open('list_patition_label.txt', 'r')
labels = label_file.readlines()

img_dir = 'RAF_DB'

emotion = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']

train_x = []
train_y = []
test_x = []
test_y = []

count = 0
idx = 0

for img in os.listdir(img_dir):
    img_number = int(img.split('_')[1].split('.')[0])
    img_array = cv2.imread(os.path.join(img_dir, img))

    faces = RetinaFace.detect_faces(img_array, threshold=0.5)

    if isinstance(faces, tuple):
        count += 1
        crop_img = img_array
        if img.split('_')[0] == 'train':
            label = labels[img_number - 1][-2]
            train_y.append(int(label)-1)
            crop_img = cv2.resize(crop_img, (48,48))
            train_x.append(crop_img)
        else:
            label = labels[img_number - 1 + 12271][-2]
            test_y.append(int(label)-1)
            crop_img = cv2.resize(crop_img, (48, 48))
            test_x.append(crop_img)
    else:
        for key in faces.keys():
            identity = faces[key]
            facial_area = identity['facial_area']

            if facial_area[3] - facial_area[1] <= 10 or facial_area[2] - facial_area[0] <= 10:
                continue
            else:
                crop_img = img_array[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]
                crop_img = cv2.resize(crop_img, (48,48))

            if img.split('_')[0] == 'train':
                idx += 1
                if idx % 100 == 0:
                    print(idx)
                label = labels[img_number - 1][-2]
                train_y.append(int(label)-1)
                train_x.append(crop_img)
            else:
                label = labels[img_number - 1 + 12271][-2]
                test_y.append(int(label)-1)
                test_x.append(crop_img)

print('unrecognized_face: ', count)
print(f"train_x shape: {train_x.shape} - train_y shape: {train_y.shape}")
print(f"test_x shape: {test_x.shape} - test_y shape: {test_y.shape}")

datafile = h5py.File(datapath, 'w')
datafile.create_dataset("training_pixel", dtype='uint8', data=train_x)
datafile.create_dataset("training_label", dtype='int64', data=train_y)
datafile.create_dataset("test_pixel", dtype='uint8', data=test_x)
datafile.create_dataset("test_label", dtype='int64', data=test_y)

datafile.close()
label_file.close()
