from retinaface import RetinaFace
import cv2
import torchvision.transforms as transforms
import torch
from PIL import Image
from torch.autograd import Variable
import glob


def fer(img):
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.536219, 0.41908908, 0.37291506], std=[0.24627768, 0.21669856, 0.20367864])
    ])

    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    _, pred = torch.max(out.data, 1)

    return pred.data.item()


if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('../checkpoints/retina/best.pth')
    model.eval()
    model.to(DEVICE)

    image_path = '../dataset/multiface/*/*'
    testList = glob.glob(image_path)

    total_face = 0
    correct_image = 0
    correct_face = 0

    for file in testList:
        count = 0
        num = 0
        img = cv2.imread(file)
        faces = RetinaFace.extract_faces(img, align=True)
        if file.split('_')[-1].split('.')[0]=='':
            print(file)
            continue
        file_class = int(file.split('_')[-1].split('.')[0])
        # if len(faces) == 0:
        #     total+=1
        #     if fer(Image.fromarray(img), file):
        #         count+=1
        # else:
        for index, face in enumerate(faces):
            if len(face[0]) <= 10 or len(face[1]) <= 10:
                continue
            else:
                label = file_class % 10
                file_class //= 10
                crop_img = Image.fromarray(face[:, :, ::-1])
                num += 1
            if fer(crop_img) == label - 1:
                count += 1
            else: print(file.split('/')[-1], fer(crop_img), label - 1)

        total_face += num
        correct_face += count
        if count == num:
            correct_image += 1
    print('Total faces: ' + str(total_face))
    print('Total image: ' + str(len(testList)))
    print('Match ratio: ' + str(correct_image / len(testList)), 'Accuracy: ' + str(correct_face / total_face))
