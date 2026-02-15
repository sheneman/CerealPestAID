#!/bin/bash
# Train MobileNetV3-Large on CerealPestAID dataset

#SBATCH -p sheneman --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=sheneman@uidaho.edu
#SBATCH --output=train_mobilenetv3.out

nvidia-smi -L
source /usr/modules/init/bash
module load cuda/11.8

source ./venv/bin/activate

python -u ./training/train_mobilenetv3.py
