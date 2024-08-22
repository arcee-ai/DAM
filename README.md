# HF Trainer for DAM

The `hf_trainer` folder contains all the code necessary to run the merging coefficient tuning process.

## Steps to Run the Workflow

### 1. Create the Merged Model
First, create the merged model by running the `merge.py` script.

### 2. Prepare the Dataset with Top-K Logits
After creating the merged model, navigate to the `data` folder inside `hf_trainer` and run `create_dataset_top_k_logits.py`. This script will:
- Collect different datasets.
- Apply their templates.
- Tokenize the data.
- Compute and save the top-K logits for other models, which will be used later during training.

### 3. Modeling
The `modeling` folder contains the modified Hugging Face classes necessary to load the DAM (Dynamic Attention Module) layers. Ensure these classes are correctly set up before proceeding to training.

### 4. Configuration Files
The `accelerate_config` folder holds all the configurations related to the training environment, distributed computing settings, and other parameters. Make sure your configuration files are correctly set before running the training.

### 5. Run the Training
Finally, you can run the training process with the `train_top_k.py` script. Use the following command:

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_config/your_yaml.yaml --num_processes 4 train_top_k.py
```

### Notes:
- Replace `your_yaml.yaml` with the actual YAML configuration file that you have set up for `accelerate`.
- The number of processes (`--num_processes`) can be adjusted based on your hardware and the specific needs of your training task.