import time
import wandb
import torch.backends.cudnn as cudnn


wandb_config = {
     'method': 'bayes',
       'metric': {
         'name': ['accuracy', 'val_epoch_loss'],
         'goal': 'maximize'   
       },
       'parameters': {
#           'layers': {
#               'values': [32, 64, 96, 128, 256]
#
#           },
           'batch_size': {
               'values': [32, 64, 96, 128]
           },
#           'epochs': {
#               'values': [5, 10, 15]
#            },
           'learning_rate':{
#               'values': [0.001,0.01,0.1]
                'values': 0.01
           }
       }
}

hyperparameter_defaults = {
        "epochs": 10,
        "learning_rate": 0.01,
        "batch_size": 32,
        }
timestr = time.strftime("%Y%m%d-%H%M%S")
cudnn_benchmark = True

#wandb.init(config=hyperparameter_defaults)
#config = wandb.config

data_dir = '../data/hands/Hands/Hands'
