# DAM Implementation

This repository contains two different implementations for running the merging coefficient tuning process:

1. **dam_with_buffer**: This implementation uses merge model buffers.
2. **dam_with_parameter_list**: This implementation utilizes a parameter list.

You can choose either implementation based on your specific requirements.

## Steps to Run the Workflow

### 1. Create the Merged Model
First, create the merged model by running the `merge.py` script found in the respective folder (`dam_with_buffer` or `dam_with_parameter_list`).

### 2. Prepare the Dataset with Top-K Logits
After creating the merged model, navigate to the `data` folder inside the chosen implementation (either `dam_with_buffer` or `dam_with_parameter_list`) and run `create_dataset_top_k_logits.py`. This script will:
- Collect different datasets.
- Apply their templates.
- Tokenize the data.
- Compute and save the top-K logits for other models, which will be used later during training.

### 3. Modeling
The `modeling` folder should be modified to include the DAM layers. To add your models:
- Place your models within the respective implementation folder (`dam_with_buffer` or `dam_with_parameter_list`).
- Ensure that the DAM layers are correctly integrated within these models before proceeding to training.

### 4. Configuration Files
The `accelerate_config` folder holds all the configurations related to the training environment, distributed computing settings, and other parameters. Make sure your configuration files are correctly set before running the training.

### 5. Run the Training
Finally, you can run the training process with the `train_top_k.py` script. Use the following command:

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_config/your_config.json --num_processes 8 train_top_k.py
```

### Important Notes:
- Replace `your_config.json` with the actual JSON configuration file that you have set up for `accelerate`.
- The number of processes (`--num_processes`) can be adjusted based on your hardware and the specific needs of your training task.
- If you decide not to use `accelerate`, DeepSpeed, or FSDP for training and prefer to use a simple training Python script, ensure that you include the following line to automatically manage device allocation:

```python
merged_model = MergedMistralForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
```