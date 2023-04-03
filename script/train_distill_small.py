import argparse
import json
# import matplotlib.pyplot as plt
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from timm.models import swinv2_base_window16_256, swinv2_small_window16_256
from timm.utils import accuracy, AverageMeter
from sklearn.metrics import classification_report
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torchvision import datasets
import torch.nn as nn

from model import create_model
from distill_loss import DistillationLoss
from util import make_dir, EMA

torch.backends.cudnn.benchmark = False
import warnings

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('Manifold Distillation', add_help=False)
    parser.add_argument('--distillation-type', default='soft', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-beta', default=1.0, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    parser.add_argument('--s-id', nargs='+', type=int)
    parser.add_argument('--t-id', nargs='+', type=int)

    parser.add_argument('--w-sample', default=0.1, type=float, help="")
    parser.add_argument('--w-patch', default=4, type=int, help="")
    parser.add_argument('--w-rand', default=0.2, type=float, help="")
    parser.add_argument('--K', default=192, type=int, help="")
    return parser


class Swin(nn.Module):
    def __init__(self, swin):
        super().__init__()
        self.swin = swin
        num_ftrs = swin.head.in_features
        self.head = nn.Linear(num_ftrs, 7)

    def forward_features(self, x):
        x = self.swin.patch_embed(x)
        if self.swin.absolute_pos_embed is not None:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.pos_drop(x)

        block_outs = []
        for layer in self.swin.layers:
            x_= x
            for blk in layer.blocks:
                x_ = blk(x_)
                block_outs.append(x_)
            x = layer(x)

        x = self.swin.norm(x)  # B L C
        return x, block_outs


    def forward(self, x):
        x, block_outs = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x, block_outs


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    # loss_base_meter= AverageMeter()
    # loss_dist_meter= AverageMeter()
    # loss_mf_sample_meter= AverageMeter()
    # loss_mf_patch_meter= AverageMeter()
    # loss_mf_rand_meter= AverageMeter()
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        if len(data) % 2 != 0:
            if len(data) < 2:
                continue
            data = data[0:len(data) - 1]
            target = target[0:len(target) - 1]
            print(len(data))
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        samples, targets = mixup_fn(data, target)
        output = model(samples)
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = criterion_train(output[0], targets)
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
            loss = criterion_train(output[0], targets)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(train.parameters(), CLIP_GRAD)
            optimizer.step()
            if use_ema and epoch % ema_epoch == 0:
                ema.update()
        torch.cuda.synchronize()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        acc1, acc5 = accuracy(output[0], target, topk=(1, 5))
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
        loss = criterion_val(output[0], target)
        _, pred = torch.max(output[0].data, 1)
        for p in pred:
            pred_list.append(p.data.item())
        acc1, acc5 = accuracy(output[0], target, topk=(1, 5))
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
    parser = argparse.ArgumentParser('Manifold KD training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    file_dir = '../checkpoints/retina_small'
    trainset_path = '../dataset/RAFDBRefined/train'
    valset_path = '../dataset/RAFDBRefined/val'
    model_name = 'base'
    make_dir(file_dir)

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
    model_path = '../checkpoints/FER/best.pth'
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
        transforms.Normalize(mean=[0.536219, 0.41908908, 0.37291506], std=[0.24627768, 0.21669856, 0.20367864]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.536219, 0.41908908, 0.37291506], std=[0.24627768, 0.21669856, 0.20367864])
    ])

    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=0.1, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=class_num)

    dataset_train = datasets.ImageFolder(trainset_path, transform=transform)
    dataset_test = datasets.ImageFolder(valset_path, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    criterion_train = SoftTargetCrossEntropy()
    criterion_val = torch.nn.CrossEntropyLoss()

    model_ft = Swin(swinv2_small_window16_256(pretrained=True))

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

        # fig = plt.figure(1)
        # plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train Loss')
        # plt.plot(epoch_list, val_loss_list, 'b-', label=u'Val Loss')
        # plt.legend(["Train Loss", "Val Loss"], loc="upper right")
        # plt.xlabel(u'epoch')
        # plt.ylabel(u'loss')
        # plt.title('Model Loss ')
        # plt.savefig(file_dir + "/loss.png")
        # plt.close(1)
        #
        # fig2 = plt.figure(2)
        # plt.plot(epoch_list, train_acc_list, 'r-', label=u'Train Acc')
        # plt.plot(epoch_list, val_acc_list, 'b-', label=u'Val Acc')
        # plt.legend(["Train Acc", "Val Acc"], loc="lower right")
        # plt.title("Model Acc")
        # plt.ylabel("acc")
        # plt.xlabel("epoch")
        # plt.savefig(file_dir + "/acc.png")
        # plt.close(2)
