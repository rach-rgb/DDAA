import torch
from torchvision import transforms


class ToFloatTensor(object):
    def __call__(self, img):
        return torch.unsqueeze(img.type(torch.FloatTensor) / 255, 0)


train_MNIST = transforms.Compose([
    ToFloatTensor(),
    transforms.Normalize(0.1307, 0.3081)
])

train_CIFAR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

valid_pre_MNIST = transforms.Compose([
    ToFloatTensor()
])

valid_after_MNIST = transforms.Compose([
    transforms.Normalize(0.1307, 0.3081)
])

valid_pre_CIFAR = transforms.Compose([
    transforms.ToTensor()
])

valid_after_CIFAR = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

test_MNIST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])

test_CIFAR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
