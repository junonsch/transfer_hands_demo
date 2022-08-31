from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchmetrics import F1Score
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import wandb
from sklearn.model_selection import KFold
import random
random.seed(41)

KFOLD = KFold(n_splits=10, shuffle=True)

from overresnet34f1.config import wandb_config

wandb.config = wandb_config

cudnn.benchmark = True

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

# cross validation


def train_epoch(model,device,dataloader,loss_fn,optimizer,exp_lr_scheduler):
    train_loss,train_correct=0.0,0
    model.train()
    for images, labels in dataloader:
        dataset_size= len(images)

        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()
        f1 = F1Score(num_classes=2).to(device)
        f1_train = f1(output, labels)
#    exp_lr_scheduler.step()

    return train_loss,train_correct, f1_train
  
def valid_epoch(model,device,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:
        dataset_size= len(images)

        images,labels = images.to(device),labels.to(device)
        output = model(images)
        loss=loss_fn(output,labels)
        valid_loss+=loss.item()*images.size(0)
        scores, predictions = torch.max(output.data,1)
        val_correct+=(predictions == labels).sum().item()
        f1 = F1Score(num_classes=2).to(device)
        f1_val = f1(output, labels)

    return valid_loss,val_correct, f1_val

def train_tune_model_cv(device, model_conv, image_datasets, 
                          loss_func, optimizer, exp_lr_scheduler, num_epochs, tuned):
    since = time.time()

    best_model_conv_wts = copy.deepcopy(model_conv.state_dict())
    best_f1 = 0.0
    
    wandb.watch(model_conv)
    alltrainloader = torch.utils.data.DataLoader(image_datasets["train"],batch_size=10)
    alltestloader = torch.utils.data.DataLoader(image_datasets["val"],batch_size=10)
    
    loss_values = []
    f1_values = []
           
    dataset = ConcatDataset([alltrainloader, alltestloader])
    for fold, (train_ids, test_ids) in enumerate(KFOLD.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')

         # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                          image_datasets["train"], 
                          batch_size=10, sampler=train_ids)
        train_dataset_size = {"train": len(train_ids)}
        testloader = torch.utils.data.DataLoader(
                          image_datasets["val"],
                          batch_size=10, sampler=test_ids)
        test_dataset_size = {"val": len(test_ids)}

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            train_loss,train_correct, f1_train = train_epoch(model_conv,device,trainloader,loss_func,optimizer,exp_lr_scheduler)  
            valid_loss, val_correct, f1_val = valid_epoch(model_conv,device,testloader,loss_func)
            train_loss = train_loss / len(trainloader.sampler)
            train_acc = train_correct / len(trainloader.sampler) * 100
            val_loss = valid_loss / len(testloader.sampler)
            val_acc = val_correct / len(testloader.sampler) * 100
            wandb.log({f"train_epoch_loss": train_loss})
            wandb.log({f"train_epoch_acc": train_acc})
            wandb.log({f"val_epoch_loss": val_loss})
            wandb.log({f"val_epoch_acc": val_acc})
            wandb.log({f"train_epoch_f1":f1_train})
            wandb.log({f"val_epoch_f1":f1_val})
            loss_values.append(val_loss)
            f1_values.append(f1_val)
            
                # deep copy the model_conv
            if f1_val > best_f1:
                best_f1 = f1_val
                best_model_conv_wts = copy.deepcopy(model_conv.state_dict())


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best test Acc: {best_acc:4f}')

    # load best model_conv weights
    model_conv.load_state_dict(best_model_conv_wts)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    if tuned:
        tuned = "tuned"
    else:
        tuned = "featex"
    torch.save(model_conv.state_dict(), f'models/f1_{tuned}_overresnet34model_conv_{timestr}.pt')
    torch.save(loss_values, f'loss_values/f1_{tuned}_overresnet34_loss_values_{timestr}.pkl')
    torch.save(f1_values, f'f1_values/f1_{tuned}_overresnet34_f1_values_{timestr}.pkl')
    wandb.finish()

    return model_conv
