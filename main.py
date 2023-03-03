from retinaface import RetinaFace
import cv2
import torchvision.transforms as transforms
import torch
import os
from PIL import Image
from torch.autograd import Variable
import glob


def fer(img):
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5449866, 0.4257745, 0.37842172],
            std=[0.21604884, 0.19581884, 0.18650503])
    ])

    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    _, pred = torch.max(out.data, 1)
    print('Image Name:{},predict:{}'.format(file, emotion[pred.data.item()]))

if __name__ == '__main__':
    emotion = ('surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral')
    label_file = open('list_patition_label.txt', 'r')
    labels = label_file.readlines()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('checkpoints/best.pth')
    model.eval()
    model.to(DEVICE)

    image_path = 'dataRefined/test/*.jpg'
    testList = glob.glob(image_path)

    idx = 0

    for file in testList:
        idx += 1
        if idx % 100 == 0:
            print(idx)

        img_number = int(file.split('_')[1].split('.')[0])
        file_class = labels[img_number - 1 + 12271][-2]

        img = cv2.imread(file)
        faces = RetinaFace.extract_faces(img, align=True)
        if len(faces)==0:
            fer(Image.fromarray(img))
        else:
            idx=0
            for key in faces.keys():
                idx+=1
                identity = faces[key]
                facial_area = identity['facial_area']
                if facial_area[3] - facial_area[1] <= 15 or facial_area[2] - facial_area[0] <= 15:
                    continue
                else:
                    crop_img = img[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]
                    crop_img = Image.fromarray(crop_img)
                fer(crop_img)

    label_file.close()
