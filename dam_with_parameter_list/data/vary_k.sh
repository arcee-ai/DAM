#!/bin/bash

# Define the values of k
k_values=(8 32 64 128)

# Loop through the k values and run the script
for k in "${k_values[@]}"
do
    echo "Running script for k = $k"
    python create_dataset_topk_logits.py --k $k
done