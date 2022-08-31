#!/bin/bash -eux

#SBATCH --job-name=run_sweep_featex_resnet34

#SBATCH --mail-type=END,FAIL

#SBATCH --mail-user=juliana.schneider@hpi.de

#SBATCH --partition=gpu

#SBATCH --cpus-per-task=8 # TODO here

#SBATCH --mem=12gb # TODO here

#SBATCH --time=12:00:00

#SBATCH --gpus=1

#SBATCH --array=0-11%4 # TODO here --> this means 12 jobs are started, at most 4 in parallel

#SBATCH -o ../sbatch_sweep_log/resnet_featex.log

#eval "$(conda shell.bash hook)"
#conda activate conda_hands2
source venv.sh


SWEEP_ID_RESNET_FEATEX=junonsch/transfer_hands-resnet34/fkxi0uzg
echo 'starting agent in sweep $SWEEP_ID_RESNET_FEATEX ...'
wandb agent --count 1 ${SWEEP_ID_RESNET_FEATEX}
echo 'finished agent in sweep $SWEEP_ID_RESNET_FEATEX .'
