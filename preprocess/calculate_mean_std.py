from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms

if __name__ == '__main__':
    data_dir = '../dataset/RAFDBRefined/train/'  # The path to the training dataset
    train_dataset = ImageFolder(root=data_dir, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()

    mean.div_(len(train_dataset))
    std.div_(len(train_dataset))
    print(f'mean={list(mean.numpy())}, std={list(std.numpy())}')
