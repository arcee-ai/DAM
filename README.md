# DAM Implementation

This repository contains two different implementations for running the merging coefficient tuning process:

1. **dam_with_buffer**: This implementation uses merge model buffers.
2. **dam_with_parameter_list**: This implementation utilizes a parameter list.

You can choose either implementation based on your specific requirements.

## Steps to Run the Workflow

### 1. Create the Merged Model
First, create the merged model by running the `merge.py` script found in the respective folder (`dam_with_buffer` or `dam_with_parameter_list`). The resulting merged model will contain untrained coefficients.

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

### 4. Run the Training
Finally, you can run the training process with the `train_top_k.py` script using the following command:

```bash
python train_topk.py
```

### Important Notes:
- The trainer will automatically divide the model across available devices. Please adjust the batch size as necessary to fit your hardware and training requirements.