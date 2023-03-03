import json
import os
import shutil

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from retinaface import RetinaFace
from timm.utils import accuracy, AverageMeter
from sklearn.metrics import classification_report
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torchvision import datasets
from timm.models.swin_transformer_v2 import swinv2_base_window16_256

torch.backends.cudnn.benchmark = False
import warnings

warnings.filterwarnings("ignore")
from ema import EMA

from retinaface import RetinaFace
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


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        if len(data) % 2 != 0:
            if len(data) < 2:
                continue
            data = data[0:len(data) - 1]
            target = target[0:len(target) - 1]
            print(len(data))
        images = []
        map_size = (256, 256)
        width, height = map_size
        for img in data:
            img = np.asarray(img).transpose(1, 2, 0)
            faces = RetinaFace.detect_faces(img)
            if type(faces) == dict:
                for face in faces:
                    identity = faces[face]

                    landmarks = identity["landmarks"]
                all_token = []
                heatmaps = []
                for key in landmarks:
                    landmark = landmarks[key]
                    heatmap = np.zeros(map_size, dtype=float)

                    x = width * landmark[0] / img.shape[0]
                    y = height * landmark[1] / img.shape[1]

                    heatmaps.append(generate_heatmap(heatmap, (x, y), 3))
                    transposed_image = img.transpose(2, 1, 0)
                    token = []
                    for channel in transposed_image:
                        token.append(heatmap * channel)
                        # token.append(np.sum(heatmap*channel))
                    all_token.append(token)

                # all_token = np.expand_dims(np.array(all_token).transpose(1, 0), axis=0)
                images.append(np.sum(np.array(all_token), axis=0))
        data = torch.tensor(images).to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        samples, targets = mixup_fn(data, target)
        output = model(samples)
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = criterion_train(output, targets)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
            if use_ema and epoch % ema_epoch == 0:
                ema.update()
        else:
            loss = criterion_train(output, targets)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(train.parameters(), CLIP_GRAD)
            optimizer.step()
            if use_ema and epoch % ema_epoch == 0:
                ema.update()
        torch.cuda.synchronize()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loss_meter.update(loss.item(), target.size(0))
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), lr))
    ave_loss = loss_meter.avg
    acc = acc1_meter.avg
    print('epoch:{}\tloss:{:.2f}\tacc:{:.2f}'.format(epoch, ave_loss, acc))
    return ave_loss, acc


@torch.no_grad()
def val(model, device, val_loader):
    global Best_ACC
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(val_loader.dataset)
    print(total_num, len(val_loader))
    val_list = []
    pred_list = []
    if use_ema and epoch % ema_epoch == 0:
        ema.apply_shadow()
    for data, target in val_loader:
        for t in target:
            val_list.append(t.data.item())
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(data)
        loss = criterion_val(output, target)
        _, pred = torch.max(output.data, 1)
        for p in pred:
            pred_list.append(p.data.item())
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
    if use_ema and epoch % ema_epoch == 0:
        ema.restore()
    acc = acc1_meter.avg
    print('\nVal set: Average loss: {:.4f}\tAcc1:{:.3f}%\tAcc5:{:.3f}%\n'.format(
        loss_meter.avg, acc, acc5_meter.avg))
    if acc > Best_ACC:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module, file_dir + "/" + 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
            torch.save(model.module, file_dir + '/' + 'best.pth')
        else:
            torch.save(model, file_dir + "/" + 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
            torch.save(model, file_dir + '/' + 'best.pth')
        Best_ACC = acc
    return val_list, pred_list, loss_meter.avg, acc

if __name__ == '__main__':
    file_dir = 'checkpoints'
    if os.path.exists(file_dir):
        print('Directory exists.')
        shutil.rmtree(file_dir)
        os.makedirs(file_dir, exist_ok=True)
    else:
        os.makedirs(file_dir)

    # set parameters
    img_size = 256
    model_lr = 1e-4
    weight_decay = 1e-8
    BATCH_SIZE = 16
    EPOCHS = 30
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = True  # Mixed precision training
    use_dp = True
    class_num = 7
    resume = False
    CLIP_GRAD = 5.0
    model_path = 'checkpoints_FER/best.pth'
    Best_ACC = 0
    use_ema = True
    ema_epoch = 32

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, expand=True),
        transforms.RandomGrayscale(p=0.25),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5916177, 0.54559606, 0.52381414], std=[0.2983136, 0.3015795, 0.30528155]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5916177, 0.54559606, 0.52381414], std=[0.2983136, 0.3015795, 0.30528155])
    ])

    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=0.1, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=class_num)

    dataset_train = datasets.ImageFolder('dataRefined/train', transform=transform)
    dataset_test = datasets.ImageFolder("dataRefined/val", transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    criterion_train = SoftTargetCrossEntropy()
    criterion_val = torch.nn.CrossEntropyLoss()

    model_ft = swinv2_base_window16_256(pretrained=True)
    num_ftrs = model_ft.head.in_features

    model_ft.head = nn.Linear(num_ftrs, class_num)

    if resume:
        model_ft = torch.load(model_path)
    model_ft.to(DEVICE)

    optimizer = optim.AdamW(model_ft.parameters(), lr=model_lr, weight_decay=weight_decay)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=2e-7)

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    if torch.cuda.device_count() > 1 and use_dp:
        print("Use", torch.cuda.device_count(), "GPUs")
        model_ft = torch.nn.DataParallel(model_ft)
    if use_ema:
        ema = EMA(model_ft, 0.999)
        ema.register()

    is_set_lr = False
    log_dir = {}
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, epoch_list = [], [], [], [], []
    for epoch in range(1, EPOCHS + 1):
        epoch_list.append(epoch)
        train_loss, train_acc = train(model_ft, DEVICE, train_loader, optimizer, epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        log_dir['train_acc'] = train_acc_list
        log_dir['train_loss'] = train_loss_list

        val_list, pred_list, val_loss, val_acc = val(model_ft, DEVICE, val_loader)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        log_dir['val_acc'] = val_acc_list
        log_dir['val_loss'] = val_loss_list
        log_dir['best_acc'] = Best_ACC

        with open(file_dir + '/result.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(log_dir))
        print(classification_report(val_list, pred_list, target_names=dataset_train.class_to_idx))

        if epoch < 600:
            cosine_schedule.step()
        else:
            if not is_set_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 1e-6
                    is_set_lr = True

        fig = plt.figure(1)
        plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train Loss')
        plt.plot(epoch_list, val_loss_list, 'b-', label=u'Val Loss')
        plt.legend(["Train Loss", "Val Loss"], loc="upper right")
        plt.xlabel(u'epoch')
        plt.ylabel(u'loss')
        plt.title('Model Loss ')
        plt.savefig(file_dir + "/loss.png")
        plt.close(1)

        fig2 = plt.figure(2)
        plt.plot(epoch_list, train_acc_list, 'r-', label=u'Train Acc')
        plt.plot(epoch_list, val_acc_list, 'b-', label=u'Val Acc')
        plt.legend(["Train Acc", "Val Acc"], loc="lower right")
        plt.title("Model Acc")
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.savefig(file_dir + "/acc.png")
        plt.close(2)
