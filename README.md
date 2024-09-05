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
python dam/merge.py mistralai/Mistral-7B-v0.1 augmxnt/shisa-gamma-7b-v1 WizardLM/WizardMath-7B-V1.1 arcee-train/Abel-7B-002-truncated-embeds --device cpu --output_path ./merged_model --use_embedding_layers --use_base_model --non_linearity tanh --merge_layernorms --repo_id arcee-train/pplist-merged-untrained-with-base-layernorm-embedding
```

#### Arguments:
- `base_model_id`: ID of the base model. All layers of this model will be replaced with DAM layers.
- `model_ids`: IDs of the models to merge. The linear layers from these models will be used.
- `--output_path`: Path where the merged model will be saved.
- `--device`: Device to use for computation (e.g., 'cpu', 'cuda').
- `--use_embedding_layers`: If specified, embedding layers will be included in the merging process.
- `--use_base_model`: If specified, trainable coefficients will also be added to the base model's linear layers. This is optional.
- `--non_linearity`: Specifies the non-linearity to use in the DAMLinearLayer. Options are `tanh`, `sigmoid`, `relu`, or `None`.
- `--merge_layernorms`: If specified, layer normalization layers will be included in the merging process.
- `--repo_id`: Repository ID where the merged model will be pushed.shed.

### 2. Prepare the Dataset

To prepare the dataset, navigate to the `dam/data` folder and run `create_merge_dataset.py`. This script will create a composite dataset with examples from the data used to train the models we are going to merge, apply their templates, and tokenize the data. Optionally, it can compute and save the top-K logits for other models, which will be used later during training. Additionally, it is optional to compute the logits beforehand; we can also compute them on-the-fly during training.

#### Command:

```bash
python dam/data/create_merge_dataset.py --k 50 --dataset_names p1atdev/ichikara-instruction:20231115-1 microsoft/orca-math-word-problems-200k meta-math/MetaMathQA --model_ids augmxnt/shisa-gamma-7b-v1 WizardLM/WizardMath-7B-V1.1 arcee-train/Abel-7B-002-truncated-embeds --base_model_name mistralai/Mistral-7B-v0.1 --cache_dir /workspace/hf-cache --compute_logits True --dataset_id arcee-train/my-combined-dataset
```

#### Arguments:
- `--k`: The number of top logits to compute and save. This is optional.
- `--dataset_names`: List of dataset names corresponding to the datasets used to tune each model. Samples will be picked from each dataset.
- `--model_ids`: List of model IDs to load.
- `--base_model_name`: Name of the base model.
- `--cache_dir`: Directory to cache the models.
- `--compute_logits`: If set to `True`, the top-K logits will be computed and saved. This is optional.
- `--dataset_id`: ID of the dataset to push to Hugging Face Hub.

### 3. Run the Training
In this step, navigate to the `dam/train_dam.py` script. The purpose of this step is to train the coefficients. At the end of the training process, the model is merged into the base model architecture with the optimized coefficients. Additionally, this code has the capability to work with multiple GPUs.

#### Command:


```bash
python dam/train_dam.py --temperature <temperature> --weight_decay <weight_decay> --learning_rate <learning_rate> --lr_scheduler_type <lr_scheduler_type> --use_kl <use_kl> --use_mse <use_mse> --use_entropy <use_entropy> --lambda_coef <lambda_coef> --lambda_coef_l1 <lambda_coef_l1> --lambda_coef_l2 <lambda_coef_l2> --generate_logits_on_fly <generate_logits_on_fly> --use_all_logits <use_all_logits> --untrained_merged_model_name <untrained_merged_model_name> --hf_disk_dataset_dir <hf_disk_dataset_dir> --cache_dir <cache_dir> --base_model_name <base_model_name>
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
- `--generate_logits_on_fly`: Generate logits on-the-fly during training.
- `--use_all_logits`: Use all logits during training.
- `--untrained_merged_model_name`: Name of the untrained merged model.
- `--hf_disk_dataset_dir`: Directory of the dataset with logits.
- `--cache_dir`: Directory to cache the models.
- `--base_model_name`: Name of the base model.

