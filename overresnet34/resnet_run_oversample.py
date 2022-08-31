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

from config import hyperparameter_defaults, timestr, cudnn_benchmark, data_dir
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

wandb.init(project=f"sweep_and_transfer_cluster_bayes")

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

from model import model_conv, loss_func, optimizer_conv, exp_lr_scheduler

model_cv = train_tune_model_cv(device, model_conv, image_datasets,
                                   loss_func, optimizer_conv, exp_lr_scheduler, num_epochs=10, tuned=False)

model_eval = evaluate_model(model_cv, class_names, dataloaders, device, loss_func)

with open(f'featex_overresnet34_model_eval_cv_{timestr}.pkl', 'wb') as handle:
    pickle.dump(model_eval, handle, protocol=pickle.HIGHEST_PROTOCOL)
