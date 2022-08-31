from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
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
from tqdm import tqdm

def evaluate_model(model_conv, class_names, dataloaders, device, loss_func):
    model_conv.eval() # Prep model for Evaluation

    mean_of = 5 # Mean of how many evaluations
    valid_loss = 0.0
    class_correct = list(0. for i in range(len(class_names))) # List of number of correct predictions in each class
    class_total = list(0. for i in range(len(class_names))) # List of total number of samples in each class
    test_accs = []
    print("First step")
    for i in tqdm(range(mean_of)):
        for data, target in dataloaders["test"]:
        # Move the data to device
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model_conv(data)
            # calculate the loss
            loss = loss_func(output, target)
            # update test loss 
            valid_loss += loss.item()*data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))

            # calculate test accuracy for each object class
            for i in range(len(target)):    
                label = target.data[i]
                if len(target) == 1:
                    class_correct[label] += correct.item()
                else:
                    class_correct[label] += correct[i].item()
                class_total[label] += 1

    # calculate and print average test loss
    valid_loss = valid_loss/(mean_of * len(dataloaders["test"].dataset))
    print('Test Loss: {:.6f}\n'.format(valid_loss))

    print("Second step")
    # print accuracy of each class
    for i in tqdm(range(len(class_names))):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %0.2f%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
            test_accs.append(100 * class_correct[i] / class_total[i])
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
    acc = 100. * np.sum(class_correct) / np.sum(class_total)

    # print total accuracy of the model
    print('\nTest Accuracy (Overall): %0.2f%% (%2d/%2d)' % (
        acc,
        np.sum(class_correct), np.sum(class_total)))

    return {"test_acc_0":test_accs[0], 
            "test_acc_1":test_accs[1],
            "test_acc_overall": acc}

