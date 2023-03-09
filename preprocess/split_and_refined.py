import cv2
from retinaface import RetinaFace
import glob
import os
import shutil
from sklearn.model_selection import train_test_split

from utils.util import make_dir


def split_data(dataset_dir, files, set_name='/train/'):
    idx = 0
    count = 0
    for img in files:
        idx += 1
        if idx % 100 == 0:
            print(idx)

        img_number = int(img.split('/')[-1].split('_')[1].split('.')[0])
        file_class = labels[img_number - 1][-2]
        file_class = os.path.join(dataset_dir + set_name, file_class)

        if not os.path.isdir(file_class):
            os.makedirs(file_class)

        count += detect_face(img, file_class)

    return count


def detect_face(img, file_class):
    img_array = cv2.imread(img)
    # faces = RetinaFace.detect_faces(img_array, threshold=0.5)

    # if isinstance(faces, tuple):
    #     count += 1
    #     shutil.copy(img, file_class + '/' + img.split('/')[1])
    # else:
    #     for key in faces.keys():
    #         identity = faces[key]
    #         facial_area = identity['facial_area']
    #
    #         if facial_area[3] - facial_area[1] <= 10 or facial_area[2] - facial_area[0] <= 10:
    #             continue
    #         else:
    #             crop_img = img_array[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]
    #             cv2.imwrite(file_class + '/' + key.split('_')[1] + '_' + img.split('/')[1], crop_img)

    faces = RetinaFace.extract_faces(img_array, align=True)

    if len(faces) == 0:
        shutil.copy(img, file_class + '/' + img.split('/')[1])
        return 1
    else:
        # for idx, face in enumerate(faces):
        # if len(face[0]) <= 10 or len(face[1]) <= 10:
        #     continue
        # cv2.imwrite(file_class + '/' + str(idx) + '_' + img.split('/')[1], face[:, :, ::-1])
        # break
        max_image_size = 0
        for idx, face in enumerate(faces):
            h = len(face[0])
            w = len(face[1])
            if max_image_size < h * w:
                max_image_size = h * w
                index = idx
        cv2.imwrite(file_class + '/' + img.split('/')[-1], faces[index][:, :, ::-1])
        return 0



if __name__ == '__main__':
    idx = 0
    count = 0

    emotion = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
    file_dir = '../dataset/RAFDBRefined'
    image_list = glob.glob('../dataset/original/RAFDB/train_*.jpg')

    label_file = open('../dataset/original/list_patition_label.txt', 'r')
    labels = label_file.readlines()

    make_dir(file_dir)

    train_files, val_files = train_test_split(image_list, test_size=0.2, random_state=42)

    test_files = glob.glob('../dataset/original/RAFDB/test_*.jpg')

    count += split_data(file_dir, train_files, set_name='/train/')
    count += split_data(file_dir, val_files, set_name='/val/')

    label_file.close()

    print('unrecognized_face: ', count)
