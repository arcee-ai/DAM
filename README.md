# Differentiable Adaptive Merging (DAM)

<img src="figure/readme.webp" alt="Project Figure" width="600"/>

Differentiable Adaptive Merging (DAM) automates the integration of multiple LLMs with unique capabilities, optimizing the balance between models for improved data efficiency and reduced computational costs. DAM outperforms traditional and evolutionary methods, offering a scalable solution for versatile AI systems. Extensive experiments validate DAM's superiority across various model merging scenarios.

## Steps to Run the Workflow

This repository contains the implementation for running the merging coefficient tuning process.

### 1. Create the Merged Model
First, create the merged model by running the `merge.py` script found in the `src` folder. The resulting merged model will contain untrained coefficients.

#### Command:


```bash
python src/merge.py <base_model_id> <model_ids> --output_path <output_path> --device <device> --use_embedding_layers --use_base_model --non_linearity <non_linearity> --merge_layernorms --repo_id <repo_id>
```

#### Arguments:
- `base_model_id`: ID of the base model (every weight except linear layers will be sourced from this model).
- `model_ids`: IDs of the models to merge (for linear layers).
- `--output_path`: Path to save the merged model.
- `--device`: Device to use for computation (e.g., 'cpu', 'cuda').
- `--use_embedding_layers`: Include embedding layers in the merging process.
- `--use_base_model`: Include base model's linear layers in the merging process.
- `--non_linearity`: Non-linearity to use in DAMLinearLayer (`tanh`, `sigmoid`, `relu`, or `None`).
- `--merge_layernorms`: Include layer normalization layers in the merging process.
- `--repo_id`: Repository ID to push the merged model to.

### 2. Prepare the Dataset (Optional)
This step is optional and depends on your chosen logits computation method:

- If you choose to compute logits on-the-fly, you can skip this step.
- If you prefer to use pre-computed logits, navigate to the `src/data` folder and run `create_dataset_topk_logits.py`. This script will:
  - Collect different datasets.
  - Apply their templates.
  - Tokenize the data.
  - Optionally compute and save the top-K logits for other models, which will be used later during training.

#### Command:


#### Command:


```bash
python src/data/create_dataset_topk_logits.py --k 50 --dataset_names p1atdev/ichikara-instruction:20231115-1 microsoft/orca-math-word-problems-200k meta-math/MetaMathQA --model_ids augmxnt/shisa-gamma-7b-v1 WizardLM/WizardMath-7B-V1.1 arcee-train/Abel-7B-002-truncated-embeds --base_model_name mistralai/Mistral-7B-v0.1 --cache_dir /workspace/hf-cache --compute_logits True
```

#### Arguments:
- `model_ids`: List of model IDs to load.
- `cache_dir`: The directory to cache the models.

### 3. Modeling
The `modeling` folder should be modified to include the DAM layers. To add your models:
- Place your models within the `src` folder.
- Ensure that the DAM layers are correctly integrated within these models before proceeding to training.

### 4. Run the Training
Finally, you can run the training process with the `train_top_k.py` script using the following command:

#### Command:


```bash
python src/train_top_k.py --temperature <temperature> --weight_decay <weight_decay> --learning_rate <learning_rate> --lr_scheduler_type <lr_scheduler_type> --use_kl <use_kl> --use_mse <use_mse> --use_entropy <use_entropy> --lambda_coef <lambda_coef> --lambda_coef_l1 <lambda_coef_l1> --lambda_coef_l2 <lambda_coef_l2> --generate_logits_on_fly <generate_logits_on_fly> --use_all_logits <use_all_logits> --untrained_merged_model_name <untrained_merged_model_name> --hf_disk_dataset_dir <hf_disk_dataset_dir> --cache_dir <cache_dir> --base_model_name <base_model_name>
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

