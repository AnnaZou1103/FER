import torch
from retinaface import RetinaFace
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
from PIL import Image
import time


def fer(img):
    emotion = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.535038, 0.41776547, 0.37159777],
            std=[0.24516706, 0.21566056, 0.20260763])
    ])

    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    _, pred = torch.max(out.data, 1)
    print(emotion[pred.data.item()])



if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('checkpoints_SwinV2_base_wopretrained/best.pth')
    model.eval()
    model.to(DEVICE)

    video_path = 'video.mp4'
    capture = cv2.VideoCapture(video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    start = time.time()
    count = 0
    while capture.isOpened():
        rval, frame = capture.read()
        count+=1
        if rval:
            faces = RetinaFace.extract_faces(frame, align=True)
            if len(faces) > 0:
                for index, face in enumerate(faces):
                    if len(face[0]) <= 10 or len(face[1]) <= 10:
                        continue
                    else:
                        crop_img = Image.fromarray(face[:, :, ::-1])
                    fer(crop_img)
        else:
            break

    end = time.time()
    seconds = end - start
    hour = int(seconds / 3600)
    minute = int(seconds % 3600 / 60)
    second = int(seconds % 60)

    print('Total frame number: '+str(frame_count))
    print('Total frame number: '+str(count))
    print('Process completed in ' + str(hour) + 'h ' + str(minute) + "m " + str(second) + 's. ')
