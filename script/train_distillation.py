import json
import matplotlib.pyplot as plt
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from timm.utils import accuracy, AverageMeter
from sklearn.metrics import classification_report
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torchvision import datasets
from model import create_model
from distill_loss import DistillationLoss
from util import make_dir

torch.backends.cudnn.benchmark = False
import warnings

warnings.filterwarnings("ignore")


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(train_loader.dataset)
    print(f'Total training samples: {total_num}, batch number: {len(train_loader)}')
    for batch_idx, (data, target) in enumerate(train_loader):
        if len(data) % 2 != 0:
            if len(data) < 2:
                continue
            data = data[0:len(data) - 1]
            target = target[0:len(target) - 1]
            print(len(data))
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        samples, targets = mixup_fn(data, target) # Apply mixup augmentation
        output = model(samples)
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                loss_base, loss_dist, loss_mf_sample, loss_mf_patch, loss_mf_rand = criterion_train(samples, output,
                                                                                                    targets)
                loss = loss_base + loss_dist + loss_mf_sample + loss_mf_patch + loss_mf_rand
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update() # Updates the scale for next iteration
        else:
            loss_base, loss_dist, loss_mf_sample, loss_mf_patch, loss_mf_rand = criterion_train(samples, output,
                                                                                                targets)
            loss = loss_base + loss_dist + loss_mf_sample + loss_mf_patch + loss_mf_rand
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        acc1, acc5 = accuracy(output[0], target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        loss_meter.update(loss_base.item(), target.size(0))
        loss_meter.update(loss_dist.item(), target.size(0))
        loss_meter.update(loss_mf_sample.item(), target.size(0))
        loss_meter.update(loss_mf_patch.item(), target.size(0))
        loss_meter.update(loss_mf_rand.item(), target.size(0))
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
    global best_acc
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(val_loader.dataset)
    print(f'Total validation samples: {total_num}, batch number:{len(val_loader)}')
    val_list = []
    pred_list = []
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
    acc = acc1_meter.avg
    print('\nVal set: Average loss: {:.4f}\tAcc1:{:.3f}%\tAcc5:{:.3f}%\n'.format(
        loss_meter.avg, acc, acc5_meter.avg))
    if acc > best_acc:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module, file_dir + "/" + 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
            torch.save(model.module, file_dir + '/' + 'best.pth')
        else:
            torch.save(model, file_dir + "/" + 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
            torch.save(model, file_dir + '/' + 'best.pth')
        best_acc = acc
    return val_list, pred_list, loss_meter.avg, acc


if __name__ == '__main__':
    file_dir = '../checkpoints/retina_dis' # Path to store the saved model and other output files
    trainset_path = '../dataset/RAFDBRefined/train' # Path to the training set
    valset_path = '../dataset/RAFDBRefined/val' # Path to the validation set
    make_dir(file_dir)

    # Set parameters
    img_size = 256
    model_lr = 1e-4
    weight_decay = 1e-8
    batch_size = 16
    epochs = 30
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = True  # Mixed precision training
    use_dp = True  # Data parallel
    class_num = 7
    resume = False  # Whether resume training
    clip_grad = 5.0  # Clip gradients
    model_path = '../checkpoints/FER/best.pth'
    best_acc = 0

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

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    teacher_model = torch.load('../checkpoints/teacher/best.pth')
    teacher_model.eval()
    teacher_model.to(device)

    criterion = SoftTargetCrossEntropy()
    criterion_train = DistillationLoss(criterion, teacher_model)

    criterion_val = torch.nn.CrossEntropyLoss()

    model_ft = create_model(model_name='distill_small')

    if resume:
        model_ft = torch.load(model_path)
    model_ft.to(device)

    optimizer = optim.AdamW(model_ft.parameters(), lr=model_lr, weight_decay=weight_decay)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=2e-7)

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    if torch.cuda.device_count() > 1 and use_dp:
        print("Use", torch.cuda.device_count(), "GPUs")
        model_ft = torch.nn.DataParallel(model_ft)

    is_set_lr = False
    log_dir = {}
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, epoch_list = [], [], [], [], []
    for epoch in range(1, epochs + 1):
        epoch_list.append(epoch)
        train_loss, train_acc = train(model_ft, device, train_loader, optimizer, epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        log_dir['train_acc'] = train_acc_list
        log_dir['train_loss'] = train_loss_list

        val_list, pred_list, val_loss, val_acc = val(model_ft, device, val_loader)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        log_dir['val_acc'] = val_acc_list
        log_dir['val_loss'] = val_loss_list
        log_dir['best_acc'] = best_acc

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

        # Save plotted figures
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
