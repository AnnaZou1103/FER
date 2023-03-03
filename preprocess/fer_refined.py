from retinaface import RetinaFace
import cv2
import os
import csv
import numpy as np
import h5py

Training_x = []
Training_y = []
PublicTest_x = []
PublicTest_y = []
PrivateTest_x = []
PrivateTest_y = []

count = 0
idx=0

file = 'fer2013.csv'
datapath = os.path.join('cropped_fer','data.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

with open(file,'r') as csvin:
    data = csv.reader(csvin)
    header = next(data)
    for row in data:
        idx += 1
        if idx % 10==0:
            print(idx)
            print('unrecognized_face: ', count)

        temp_list = []
        for pixel in row[1].split():
            temp_list.append([int(pixel), int(pixel), int(pixel)])

        I = np.asarray(temp_list)
        I = np.reshape(I, (48,48, 3)).astype('float32')

        faces = RetinaFace.detect_faces(I, threshold=0.5)

        if isinstance(faces, tuple):
            crop_img = I[:,:,0].flatten().astype('int')
            count+=1
        else:
            for key in faces.keys():
                identity = faces[key]
                facial_area = identity['facial_area']

                if facial_area[3] - facial_area[1] <= 10 or facial_area[2] - facial_area[0] <= 10:
                    crop_img = I[:,:,0].flatten().astype('int')
                    count += 1
                else:
                    crop_img = I[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]
                    # img_array = crop_img.flatten()
                    crop_img = cv2.resize(crop_img, (48, 48))
                    crop_img = crop_img[:,:,0].flatten().astype('int')
                    # cv2.imwrite('cropped_fer/' + str(idx) + '.jpg', crop_img)

        if row[-1] == 'Training':
            Training_y.append(int(row[0]))
            Training_x.append(crop_img.tolist())

        if row[-1] == "PublicTest" :
            PublicTest_y.append(int(row[0]))
            PublicTest_x.append(crop_img.tolist())

        if row[-1] == 'PrivateTest':
            PrivateTest_y.append(int(row[0]))
            PrivateTest_x.append(crop_img.tolist())

print('unrecognized_face: ', count)
print(np.shape(Training_x))
print(np.shape(PublicTest_x))
print(np.shape(PrivateTest_x))

datafile = h5py.File(datapath, 'w')
datafile.create_dataset("Training_pixel", dtype = 'uint8', data=Training_x)
datafile.create_dataset("Training_label", dtype = 'int64', data=Training_y)
datafile.create_dataset("PublicTest_pixel", dtype = 'uint8', data=PublicTest_x)
datafile.create_dataset("PublicTest_label", dtype = 'int64', data=PublicTest_y)
datafile.create_dataset("PrivateTest_pixel", dtype = 'uint8', data=PrivateTest_x)
datafile.create_dataset("PrivateTest_label", dtype = 'int64', data=PrivateTest_y)
datafile.close()