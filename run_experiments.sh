#!/bin/bash

# Experiment 1: Running with default parameters
echo "Experiment 1: Running with default parameters: KL + similarity."
python dam/train_dam.py \

echo "Experiment 2: KL Only."
python dam/train_dam.py \
    --similarity=False

echo "Experiment 3: Enabling 'l1_l2_reg' loss function."
python dam/train_dam.py \
    --l1_l2_reg=True 

echo "Experiment 4: Replace 'similarity' with 'overlap' loss function."
python dam/train_dam.py \
    --overlap=True \
    --similarity=False

echo "Experiment 5: Replacing 'kl' with 'mse'."
python dam/train_dam.py \
    --kl=False \
    --mse=True

echo "Experiment 6: Replacing 'kl' with 'entropy'."
python dam/train_dam.py \
    --kl=False \
    --entropy=True