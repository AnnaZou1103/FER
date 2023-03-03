import cv2
from retinaface import RetinaFace
import glob
import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

def generate_heatmap(heatmap, point, sigma, label_type='Gaussian'):
    """This function is borrowed from the official implementation. We
    will use the method whatever the HRNet authors used. But some
    variables are re-named to make it easier to read. Maybe someday I
    will re-write this.
    """
    # Check that any part of the gaussian is in-bounds
    tmp_size = sigma * 3
    ul = [int(point[0] - tmp_size), int(point[1] - tmp_size)]
    br = [int(point[0] + tmp_size + 1), int(point[1] + tmp_size + 1)]
    if (ul[0] >= heatmap.shape[1] or ul[1] >= heatmap.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return heatmap

    # Generate gaussian
    size_heat = 2 * tmp_size + 1
    x = np.arange(0, size_heat, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size_heat // 2

    # The gaussian is not normalized, we want the center value to equal 1
    if label_type == 'Gaussian':
        heat = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) /
                      (2 * sigma ** 2))
    else:
        heat = sigma / (((x - x0) ** 2 + (y - y0)
                         ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    x_heat = max(0, -ul[0]), min(br[0], heatmap.shape[1]) - ul[0]
    y_heat = max(0, -ul[1]), min(br[1], heatmap.shape[0]) - ul[1]

    # Image range
    x_map = max(0, ul[0]), min(br[0], heatmap.shape[1])
    y_map = max(0, ul[1]), min(br[1], heatmap.shape[0])

    heatmap[y_map[0]:y_map[1], x_map[0]:x_map[1]
    ] = heat[y_heat[0]:y_heat[1], x_heat[0]:x_heat[1]]

    return heatmap


def detect_face(count, img, file_class):
    map_size = (256,256)
    width, height = map_size
    img_array = cv2.imread(img)
    img_array = cv2.resize(img_array, map_size)
    faces = RetinaFace.detect_faces(img_array)

    if type(faces) == dict:
        for face in faces:
            identity = faces[face]
            landmarks = identity["landmarks"]
            break

        all_token=[]
        heatmaps = []
        for key in landmarks:
            landmark = landmarks[key]
            heatmap = np.zeros(map_size, dtype=float)

            x = width * landmark[0] / img_array.shape[0]
            y = height * landmark[1] / img_array.shape[1]

            heatmaps.append(generate_heatmap(heatmap, (x, y), 3))
            transposed_image = img_array.transpose(2, 1, 0)
            token = []
            for channel in transposed_image:
                token.append(heatmap * channel)
                # token.append(np.sum(heatmap*channel))
            all_token.append(token)

        # all_token = np.expand_dims(np.array(all_token).transpose(1, 0), axis=0)
        shutil.copy(np.sum(np.array(all_token), axis=0), file_class + '/' + img.split('/')[1])
    else:
        count+=1
        shutil.copy(img, file_class + '/' + img.split('/')[1])

    return count

idx = 0
count = 0

emotion = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
file_dir = 'dataRefined'
image_list = glob.glob('RAFDB/train_*.jpg')

label_file = open('list_patition_label.txt', 'r')
labels = label_file.readlines()

if os.path.exists(file_dir):
    print('Directory exists')
    shutil.rmtree(file_dir)
    os.makedirs(file_dir)
else:
    os.makedirs(file_dir)

train_files, val_files = train_test_split(image_list, test_size=0.2, random_state=42)

for img in train_files:
    idx += 1
    if idx % 100 == 0:
        print(idx)

    img_number = int(img.split('_')[1].split('.')[0])
    file_class = labels[img_number - 1][-2]
    file_class = os.path.join(file_dir + '/train/', file_class)

    if not os.path.isdir(file_class):
        os.makedirs(file_class)

    count = detect_face(count, img, file_class)

for img in val_files:
    idx += 1
    if idx % 100 == 0:
        print(idx)

    img_number = int(img.split('_')[1].split('.')[0])
    file_class = labels[img_number - 1][-2]
    file_class = os.path.join(file_dir + '/val/', file_class)

    if not os.path.isdir(file_class):
        os.makedirs(file_class)

    count = detect_face(count, img, file_class)

label_file.close()

print('unrecognized_face: ', count)
