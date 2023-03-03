import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from timm.data import create_transform

from RAFDB import RAFDB

CUDA = torch.cuda.is_available()

DEVICE = torch.device("cuda" if CUDA else "cpu")

resnet_data_transforms = {
    # 'train': transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.RandomRotation(10),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5916177, 0.54559606, 0.52381414],
    #     std=[0.2983136, 0.3015795, 0.30528155]),
    # ]),
    'train': create_transform(
        input_size=256,
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        mean=[0.5916177, 0.54559606, 0.52381414],
        std=[0.2983136, 0.3015795, 0.30528155]
    ),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5916177, 0.54559606, 0.52381414],
        std=[0.2983136, 0.3015795, 0.30528155]),
    ]),
}

continue_train = False
epoches = 10
batch_size = 20
class_num = 7 #the number of the emotion types

torch.backends.cudnn.benchmark = True

train_set = RAFDB(split='Training', transform=resnet_data_transforms['train'])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
test_set = RAFDB(split='Test', transform=resnet_data_transforms['val'])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

train_dataset_size = len(train_set)
val_dataset_size = len(test_set)

print('train_dataset_size', train_dataset_size)
print('val_dataset_size', val_dataset_size)

model_ft = models.resnet18(pretrained=True)

num_fc_ftr = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_fc_ftr, class_num)
model_ft = model_ft.to(DEVICE)

# define loss function
criterion = nn.CrossEntropyLoss()
# set different learning rate for revised fc layer and previous layers
lr = 0.0001
fc_params = list(map(id, model_ft.fc.parameters()))
base_params = filter(lambda p: id(p) not in fc_params, model_ft.parameters())
optimizer = torch.optim.Adam([{'params': base_params},
                              {'params': model_ft.fc.parameters(), 'lr': lr * 10}], lr=lr, betas=(0.9, 0.999))
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

device = DEVICE
train_acc = []
val_acc = []
best_model = model_ft.state_dict()
best_acc = 0

for epoch in range(epoches):
    model_ft.train()
    iteration = 0
    train_correct = 0
    for batch_idx, data in enumerate(train_loader):
        x, y = data
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        y_hat = model_ft(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        iteration += 1
        # get the index of the max log-probability
        pred = y_hat.max(1, keepdim=True)[1]
        train_correct += pred.eq(y.view_as(pred)).sum().item()

    print('Epoch', epoch, 'Train accuracy', train_correct/train_dataset_size)
    train_acc.append(train_correct/train_dataset_size)

    model_ft.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x, y = data
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            y_hat = model_ft(x)
            test_loss += criterion(y_hat, y).item()  # sum up batch loss
            # get the index of the max log-probability
            pred = y_hat.max(1, keepdim=True)[1]
            test_correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = test_correct / val_dataset_size
    if acc > best_acc:
        best_acc = acc
        best_model = model_ft.state_dict()
    val_acc.append(acc)

    scheduler.step()
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, len(test_set), 100. * acc))

best_model = model_ft.state_dict()

torch.save(model_ft.state_dict(), os.path.join('./checkpoints', "resnet_best.pkl"))
