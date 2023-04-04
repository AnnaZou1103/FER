import torch
from retinaface import RetinaFace
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
from PIL import Image
import time
import numpy as np
import math

model_path = 'checkpoints/retina_small/best.pth'  # The path to the stored model
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL = torch.load(model_path, map_location=DEVICE)
MODEL.eval()
MODEL.to(DEVICE)

# from timm.models import swinv2_small_window16_256
# import torch.nn as nn
#
# MODEL = swinv2_small_window16_256(pretrained=True)
# num_ftrs = MODEL.head.in_features
# MODEL.head = nn.Linear(num_ftrs, 7)
# MODEL.eval()
# MODEL.to(DEVICE)


def find_euclidean_distance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def alignment_procedure(img, left_eye, right_eye, nose):
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    center_eyes = (int((left_eye_x + right_eye_x) / 2), int((left_eye_y + right_eye_y) / 2))

    if False:
        img = cv2.circle(img, (int(left_eye[0]), int(left_eye[1])), 2, (0, 255, 255), 2)
        img = cv2.circle(img, (int(right_eye[0]), int(right_eye[1])), 2, (255, 0, 0), 2)
        img = cv2.circle(img, center_eyes, 2, (0, 0, 255), 2)
        img = cv2.circle(img, (int(nose[0]), int(nose[1])), 2, (255, 255, 255), 2)

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    a = find_euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = find_euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = find_euclidean_distance(np.array(right_eye), np.array(left_eye))

    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b * b + c * c - a * a) / (2 * b * c)

        cos_a = min(1.0, max(-1.0, cos_a))

        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

        if center_eyes[1] > nose[1]:
            img = Image.fromarray(img)
            img = np.array(img.rotate(180))

    return img


def fer(img):
    emotion = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.536219, 0.41908908, 0.37291506],
            std=[0.24627768, 0.21669856, 0.20367864])
    ])

    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = MODEL(img)
    _, pred = torch.max(out.data, 1)
    return emotion[pred.data.item()]


def detect_face(image):
    results = RetinaFace.detect_faces(image)
    if type(results) == dict:
        for key in results:
            identity = results[key]
            facial_area = identity["facial_area"]
            if facial_area[3] - facial_area[1] <= 10 or facial_area[2] - facial_area[0] <= 10:
                continue
            else:
                facial_img = image[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
                w = facial_area[2] - facial_area[0]
                h = facial_area[3] - facial_area[1]

                landmarks = identity["landmarks"]
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]
                facial_img = alignment_procedure(facial_img, right_eye, left_eye, nose)

                label = fer(Image.fromarray(facial_img))
                cv2.rectangle(image, (facial_area[0], facial_area[1]),
                              (facial_area[0] + w, facial_area[1] + h), (36, 255, 12), 1)
                cv2.putText(image, label, (facial_area[0], facial_area[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (36, 255, 12), 2)
    return image


def process_file(file_path, save_path, input_type='Video'):
    start = time.time()
    if input_type == 'Video':
        capture = cv2.VideoCapture(file_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = capture.get(cv2.CAP_PROP_FPS)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(save_path + file_path.split('/')[-1], cv2.VideoWriter_fourcc(*'mp4v'), frame_rate,
                              (frame_width, frame_height))
        while capture.isOpened():
            rval, frame = capture.read()
            if rval:
                frame = detect_face(frame)
                out.write(frame)
            else:
                break

        print('Total frame number: ' + str(frame_count))
    elif input_type == 'Image':
        img = cv2.imread(file_path)
        img = detect_face(img)
        cv2.imwrite(save_path + file_path.split('/')[-1], img)
        print(save_path + file_path.split('/')[-1])

    end = time.time()
    seconds = end - start
    hour = int(seconds / 3600)
    minute = int(seconds % 3600 / 60)
    second = int(seconds % 60)

    print('Process completed in ' + str(hour) + 'h ' + str(minute) + "m " + str(second) + 's. ')


# if __name__ == '__main__':
#     file_path = '../dataset/video.mp4'
#     save_path = '../output/processed_media/'
#     process_file(file_path, save_path, input_type='Video')
