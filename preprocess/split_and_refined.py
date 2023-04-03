import cv2
from retinaface import RetinaFace
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
    faces = RetinaFace.extract_faces(img_array, align=True)

    if len(faces) == 0:  # No face is detected
        shutil.copy(img, file_class + '/' + img.split('/')[-1])
        return 1
    else:  # Select the face with the largest coverage area
        index = 0
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
    unrecognized_face = 0

    emotion = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
    out_dir = '../dataset/RAFDBRefined' # The path to the folder storing the spilt dataset
    image_list = glob.glob('../dataset/RAFDB/original/train_*.jpg')

    label_file = open('../dataset/RAFDB/list_patition_label.txt', 'r')
    labels = label_file.readlines()

    make_dir(out_dir)

    train_files, val_files = train_test_split(image_list, test_size=0.2, random_state=42)

    test_files = glob.glob('../dataset/RAFDB/original/test_*.jpg')

    unrecognized_face += split_data(out_dir, train_files, set_name='/train/')
    unrecognized_face += split_data(out_dir, val_files, set_name='/val/')

    label_file.close()

    print(f'Unrecognized face: {unrecognized_face}')
