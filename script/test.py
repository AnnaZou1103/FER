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


def fer(image, file, contrastive=False):
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.536219, 0.41908908, 0.37291506],
            std=[0.24627768, 0.21669856, 0.20367864])
    ])

    img_number = int(file.split('/')[-1].split('_')[1].split('.')[0])
    file_class = labels[img_number - 1 + 12271][-2]

    image = transform_test(image)
    image.unsqueeze_(0)
    image = Variable(image).to(DEVICE)

    if not contrastive:
        out = model(image)
    else:
        out,feats = model(image)
    _, pred = torch.max(out.data, 1)

    result[int(file_class) - 1][pred.data.item()] += 1

    return pred.data.item() == int(file_class) - 1


if __name__ == '__main__':
    output_dir = '../output/'

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('../checkpoints/retina/best.pth')
    model.eval()
    model.to(DEVICE)

    image_path = '../dataset/RAFDBRefined/test/*.jpg'
    testList = glob.glob(image_path)
    label_file = open('../dataset/RAFDB/list_patition_label.txt', 'r')
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
