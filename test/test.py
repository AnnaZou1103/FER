from retinaface import RetinaFace
import cv2
import torchvision.transforms as transforms
import torch
import os
from PIL import Image
from torch.autograd import Variable
import glob
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

class Swin(nn.Module):
    def __init__(self, swin):
        super().__init__()
        self.swin = swin
        num_ftrs = swin.head.in_features
        self.head = nn.Linear(num_ftrs, 7)

    def forward(self, x):
        feats = self.swin.forward_features(x)
        feats = feats.mean(dim=1)
        x = self.head(feats)
        return feats, x

def fer(img, file):
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.535038, 0.41776547, 0.37159777],
            std=[0.24516706, 0.21566056, 0.20260763])
    ])

    img_number = int(file.split('_')[1].split('.')[0])
    file_class = labels[img_number - 1 + 12271][-2]

    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    feats, out = model(img)
    _, pred = torch.max(out.data, 1)

    result[int(file_class) - 1][pred.data.item()] += 1

    return pred.data.item() == int(file_class) - 1


if __name__ == '__main__':
    label_file = open('list_patition_label.txt', 'r')
    labels = label_file.readlines()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('checkpoints/best.pth')
    model.eval()
    model.to(DEVICE)

    image_path = 'dataRefined/test/*.jpg'
    testList = glob.glob(image_path)

    idx = 0
    count = 0

    result = [[0 for x in range(7)] for y in range(7)]

    for file in testList:
        idx += 1
        if idx % 100 == 0:
            print(idx)

        img = cv2.imread(file)
        faces = RetinaFace.extract_faces(img, align=True)
        if len(faces) == 0:
            if fer(Image.fromarray(img), file):
                count+=1
        else:
            for index, face in enumerate(faces):
                if len(face[0]) <= 10 or len(face[1]) <= 10:
                    continue
                else:
                    crop_img = Image.fromarray(face[:, :, ::-1])

                if fer(crop_img, file):
                    count+=1
                    break

    print('Accuracy:'+str(count/len(testList)))
    label_file.close()

    classes = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
    figure, ax = plot_confusion_matrix(conf_mat=np.array(result),
                                       class_names=classes,
                                       show_absolute=False,
                                       show_normed=True,
                                       colorbar=True)
    figure.set_size_inches(8, 8)
    plt.savefig("cm.png")
