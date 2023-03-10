from retinaface import RetinaFace
import cv2
import torchvision.transforms as transforms
import torch
from PIL import Image
from torch.autograd import Variable
import glob
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x


class BasicLayerSE(nn.Module):
    def __init__(self, dim, layer):
        super(BasicLayerSE, self).__init__()
        self.layer = layer
        self.se_layer = SELayer(dim)

    def forward(self, x):
        for blk in self.layer.blocks:
            if self.layer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.se_layer(x)
        x = self.layer.downsample(x)
        return x


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

def fer(image, file, contrastive=False):
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.535038, 0.41776547, 0.37159777],
            std=[0.24516706, 0.21566056, 0.20260763])
    ])

    img_number = int(file.split('/')[-1].split('_')[1].split('.')[0])
    file_class = labels[img_number - 1 + 12271][-2]

    image = transform_test(image)
    image.unsqueeze_(0)
    image = Variable(image).to(DEVICE)

    if not contrastive:
        out = model(image)
    else:
        feats, out = model(image)
    _, pred = torch.max(out.data, 1)

    result[int(file_class) - 1][pred.data.item()] += 1

    return pred.data.item() == int(file_class) - 1


if __name__ == '__main__':
    output_dir = '../output/'

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('../checkpoints/retina_center/best.pth')
    model.eval()
    model.to(DEVICE)

    image_path = '../dataset/RAFDBRefined/test/*.jpg'
    testList = glob.glob(image_path)
    label_file = open('../dataset/original/list_patition_label.txt', 'r')
    labels = label_file.readlines()

    idx = 0
    count = 0
    contrastive = True

    result = [[0 for x in range(7)] for y in range(7)]

    for file in testList:
        idx += 1
        if idx % 100 == 0:
            print(idx)

        img = cv2.imread(file)
        faces = RetinaFace.extract_faces(img, align=True)
        if len(faces) == 0:
            if fer(Image.fromarray(img), file, contrastive):
                count += 1
        else:
            for index, face in enumerate(faces):
                if len(face[0]) <= 10 or len(face[1]) <= 10:
                    continue
                else:
                    crop_img = Image.fromarray(face[:, :, ::-1])

                if fer(crop_img, file, contrastive):
                    count += 1
                    break

    print('Accuracy:' + str(count / len(testList)))
    label_file.close()

    classes = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
    figure, ax = plot_confusion_matrix(conf_mat=np.array(result),
                                       class_names=classes,
                                       show_absolute=False,
                                       show_normed=True,
                                       colorbar=True)
    figure.set_size_inches(8, 8)
    plt.savefig(output_dir + "w_confusion_matrix.png")
