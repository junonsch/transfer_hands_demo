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
from resnet34.config import hyperparameter_defaults
from resnet34.train import DATA_TRANSFORMS, train_tune_model_cv 
import wandb
import random
random.seed(41)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_conv = torchvision.models.resnet34(pretrained=True)

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)
model_conv = model_conv.to(device)

loss_func = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.da
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=hyperparameter_defaults["learning_rate"], momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
