from torchvision import datasets, models, transforms
import torch.nn as nn


DATA_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
       transforms.RandomPerspective(distortion_scale=0.4),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406],
[0.229, 0.224, 0.225])]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

loss_func = nn.CrossEntropyLoss()
