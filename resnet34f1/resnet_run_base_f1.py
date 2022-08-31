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
import wandb
import random
random.seed(41)


#wandb.config = {
#  "learning_rate": 0.001,
#  "epochs": 10,
#}
#wandb.config = {
#     'method': 'bayes',
#       'metric': {
#         'name': 'accuracy',
#         'goal': 'maximize'   
#       },
#       'parameters': {
#           'layers': {
#               'values': [32, 64, 96, 128, 256]#
#
 #          },
 #          'batch_size': {
 #              'values': [32, 64, 96, 128]
#
#           },
#           'epochs': {
#               'values': [5, 10, 15]
#            },
#           'learning_rate':{
#               'values': [0.001,0.01,0.1]
#           }
#       }
#}
#timestr = time.strftime("%Y%m%d-%H%M%S")
#wandb.init(project=f"hand_transfer_cluster_{timestr}_{wandb.config['learning_rate']}")

#cudnn.benchmark = True

#data_dir = './data/hands/Hands/Hands'

from config import hyperparameter_defaults, timestr, cudnn_benchmark, data_dir
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

wandb.init(project=f"sweep_and_transfer_cluster_bayes")
#wandb.config = wandb_config

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          DATA_TRANSFORMS[x])
                  for x in ['train', 'val', 'test']}


def weighted_sampling(data):
    labels = [y[1] for y in data.imgs]
    labels_unique, counts = np.unique(labels, return_counts = True)
    class_weights = [sum(counts) / c for c in counts]
    example_weights = [class_weights[e] for e in labels]
    sampler = WeightedRandomSampler(example_weights, len(labels))
    
    return sampler

train_sampler = weighted_sampling(image_datasets["train"])
#test_sampler = weighted_sampling(image_datasets["test"])
#val_sampler = weighted_sampling(image_datasets["val"])

samplers = {"train": train_sampler,
           "val": None,#val_sampler,
           "test": None} #test_sampler}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=hyperparameter_defaults["batch_size"],
                                             sampler=samplers[x], num_workers=4)
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not device:
    print("no device")

#model_conv = torchvision.models.resnet18(pretrained=True)
#for param in model_conv.parameters():
#    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
#num_ftrs = model_conv.fc.in_features
#model_conv.fc = nn.Linear(num_ftrs, 2)
#model_conv = model_conv.to(device)

#loss_func = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.da
#optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

from model import model_conv, loss_func, optimizer_conv, exp_lr_scheduler


#model_conv = train_model(device, model_conv, dataloaders,dataset_sizes, loss_func, optimizer_conv,
#                         exp_lr_scheduler, num_epochs=10)
model_cv = train_tune_model_cv(device, model_conv, image_datasets,
                                   loss_func, optimizer_conv, exp_lr_scheduler, num_epochs=10, tuned=False)

model_eval = evaluate_model(model_cv, class_names, dataloaders, device, loss_func)

with open(f'f1_featex_resnet34_model_eval_cv_{timestr}.pkl', 'wb') as handle:
    pickle.dump(model_eval, handle, protocol=pickle.HIGHEST_PROTOCOL)
