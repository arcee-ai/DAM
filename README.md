# Differentiable Adaptive Merging (DAM)

<img src="figures/readme.webp" alt="Project Figure" width="600"/>

Differentiable Adaptive Merging (DAM) automates the merging of multiple LLMs with unique capabilities, optimizing the balance between models for improved data efficiency and reduced computational costs. DAM outperforms traditional and evolutionary methods, offering a scalable solution for versatile AI systems. Extensive experiments validate DAM's superiority across various model merging scenarios.

## Steps to Run the Workflow

This repository contains the implementation for running the merging coefficient tuning process.

### 1. Create the Merged Model
First, create the merged model by running the `merge.py` script found in the `dam` folder. The resulting merged model will contain untrained coefficients.

In this step, we assign a trainable coefficient for each column of each model's layer norms, embedding layers, and linear layers as specified by the user. These coefficients will be optimized during the training process to achieve the best integration of the models.

#### Command:


```bash
python dam/merge.py mistralai/Mistral-7B-v0.1 augmxnt/shisa-gamma-7b-v1 WizardLM/WizardMath-7B-V1.1 arcee-train/Abel-7B-002-truncated-embeds --device cpu --output_path ./merged_model --merge_embedding_layers --merge_layernorms --use_base_model --non_linearity "None"  --repo_id arcee-train/pplist-merged-untrained-with-base-layernorm-embedding
```

#### Arguments:
- `base_model_id`: ID of the base model. All layers of this model will be replaced with DAM layers.
- `model_ids`: IDs of the models to merge. The linear layers from these models will be used.
- `--output_path`: Path where the merged model will be saved.
- `--device`: Device to use for computation (e.g., 'cpu', 'cuda').
- `--merge_embedding_layers`: If specified, embedding layers will be included in the merging process.
- `--merge_layernorms`: If specified, layer normalization layers will be included in the merging process.
- `--use_base_model`: If specified, trainable coefficients will also be added to the base model's linear layers. This is optional.
- `--non_linearity`: Specifies the non-linearity to use in the DAMLinearLayer. Options are `tanh`, `sigmoid`, `relu`, or `None`.
- `--repo_id`: Repository ID where the merged model will be pushed.shed.

### 2. Prepare the Dataset

To prepare the dataset, navigate to the `dam/data` folder and run `create_merge_dataset.py`. This script will create a composite dataset with examples from the data used to train the models we are going to merge, apply their templates, and tokenize the data. Optionally, it can compute and save the top-K logits for other models, which will be used later during training. Additionally, it is optional to compute the logits beforehand; we can also compute them on-the-fly during training.

#### Command:

```bash
python dam/data/create_merge_dataset.py --k 50 --dataset_names "p1atdev/ichikara-instruction:20231115-1,microsoft/orca-math-word-problems-200k,meta-math/MetaMathQA" --model_ids "augmxnt/shisa-gamma-7b-v1,WizardLM/WizardMath-7B-V1.1,arcee-train/Abel-7B-002-truncated-embeds" --base_model_name mistralai/Mistral-7B-v0.1 --cache_dir /home/ec2-user/.cache/huggingface --compute_logits True --dataset_id arcee-train/my-combined-dataset --base_model_dataset_name reflex-ai/fineweb-ultra-mini --example_count 1729 --max_length 2048 --add_top_k_logits  False
```

#### Arguments:
- `--k`: The number of top logits to compute and save. This is optional.
- `--dataset_names`: List of dataset names corresponding to the datasets used to tune each model. Samples will be picked from each dataset.
- `--base_model_dataset_name`: Name of the base model dataset. This is optional.
- `--model_ids`: List of model IDs to load.
- `--base_model_name`: Name of the base model.
- `--cache_dir`: Directory to cache the models.
- `--compute_logits`: If set to `True`, the top-K logits will be computed and saved. This is optional.
- `--dataset_id`: ID of the dataset to push to Hugging Face Hub.
- `--example_count`: Number of examples to select from each dataset.
- `--max_length`: Maximum length of the tokenized examples.
- `--add_top_k_logits`: Add top-K logits to the combined dataset. The default is False.


### 3. Run the Training
In this step, navigate to the `dam/train_dam.py` script. The purpose of this step is to train the coefficients. At the end of the training process, the model is merged into the base model architecture with the optimized coefficients. Additionally, this code has the capability to work with multiple GPUs.

Manual configurations are available at the top of the train_dam.py script. 
- Individual components of the loss function can be toggled by setting their associated value to True/False in the loss_fns dictionary 

#### Command:


```bash
python dam/train_dam.py --temperature 2.0 --weight_decay 0.01 --learning_rate 1e-2 --lr_scheduler_type linear --lambda_coef_similarity 0.01 --lambda_coef_l1 1e-6 --lambda_coef_l2 1e-5 --generate_logits_on_fly True --use_all_logits True --untrained_merged_model_name arcee-train/merged-untrained --combined_hf_dataset_dir arcee-train/my-combined-dataset --cache_dir /home/ec2-user/.cache/huggingface --base_model_name mistralai/Mistral-7B-v0.1 --use_wandb True

```

#### Arguments:
- `--temperature`: Temperature for KL divergence.
- `--weight_decay`: Weight decay for the optimizer.
- `--learning_rate`: Learning rate for the optimizer.
- `--lr_scheduler_type`: Type of learning rate scheduler (`linear`, etc.).
- `--use_kl`: Use KL divergence in the loss function.
- `--use_mse`: Use Mean Squared Error in the loss function.
- `--use_entropy`: Use entropy in the loss function.
- `--lambda_coef`: Lambda coefficient for regularization.
- `--lambda_coef_l1`: L1 regularization coefficient.
- `--lambda_coef_l2`: L2 regularization coefficient.
- `--use_wandb`: Upload training logs to Weights and Biases
- `--generate_logits_on_fly`: Generate logits on-the-fly during training.
- `--use_all_logits`: Use all logits during training.
- `--untrained_merged_model_name`: Name of the untrained merged model.
- `--combined_hf_dataset_di`: Directory of the dataset with logits.
- `--cache_dir`: Directory to cache the models.
- `--base_model_name`: Name of the base model.