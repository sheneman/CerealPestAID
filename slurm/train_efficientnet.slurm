#!/bin/bash
# Train EfficientNet-B6 on CerealPestAID dataset

#SBATCH -p vasdekis --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=sheneman@uidaho.edu
#SBATCH --output=train_efficientnet.out

nvidia-smi -L
source /usr/modules/init/bash
module load cuda/11.8

source ./venv/bin/activate

python -u ./training/train_efficientnet.py
