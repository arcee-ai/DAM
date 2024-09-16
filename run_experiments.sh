#!/bin/bash

# Experiment 1: Running with default parameters
echo "Experiment 1: Running with default parameters: KL + similarity."
python train.py \

echo "Experiment 2: KL Only."
python train.py \
    --similarity=False

echo "Experiment 3: Enabling 'l1_l2_reg' loss function."
python train.py \
    --l1_l2_reg=True 

echo "Experiment 4: Replace 'similarity' with 'overlap' loss function."
python train.py \
    --overlap=True \
    --similarity=False

echo "Experiment 5: Replacing 'kl' with 'mse'."
python train.py \
    --kl=False \
    --mse=True

echo "Experiment 6: Replacing 'kl' with 'entropy'."
python train.py \
    --kl=False \
    --entropy=True