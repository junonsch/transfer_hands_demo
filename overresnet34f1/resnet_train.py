from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle
from train import DATA_TRANSFORMS, train_tune_model_cv
from test import evaluate_model
import random
random.seed(41)
import wandb

wandb.init(project=f"train_hypertuned_overresnet34f1")


from config import timestr, cudnn_benchmark, data_dir

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          DATA_TRANSFORMS[x]) if x != "test" else datasets.ImageFolder(os.path.join(data_dir, x))
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,#4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not device:
    print("no device")


from model import model_conv, loss_func, optimizer_conv, exp_lr_scheduler


model_cv = train_tune_model_cv(device, model_conv, image_datasets,
                                   loss_func, optimizer_conv, exp_lr_scheduler, num_epochs=10, tuned=False)

torch.save(model_cv.state_dict(), f'models/TRAINED_{tuned}_f1_overresnet34model_conv_{timestr}.pt')
