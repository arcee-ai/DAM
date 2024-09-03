# DAM Implementation

<img src="figure/readme.webp" alt="Project Figure" width="600"/>

This repository contains two different implementations for running the merging coefficient tuning process:

1. **dam_with_buffer**: This implementation uses merge model buffers.
2. **dam_with_parameter_list**: This implementation utilizes a parameter list.

You can choose either implementation based on your specific requirements.

## Steps to Run the Workflow

### 1. Create the Merged Model
First, create the merged model by running the `merge.py` script found in the respective folder (`dam_with_buffer` or `dam_with_parameter_list`). The resulting merged model will contain untrained coefficients.

### 2. Prepare the Dataset (Optional)
This step is optional and depends on your chosen logits computation method:

- If you choose to compute logits on-the-fly, you can skip this step.
- If you prefer to use pre-computed logits, navigate to the `data` folder inside the chosen implementation (either `dam_with_buffer` or `dam_with_parameter_list`) and run `create_dataset_top_k_logits.py`. This script will:
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

```
python train_top_k.py --config config.yaml
```

## Seamless Switch Between On-the-Fly and Pre-Computed Logits

### Description
This enhancement to the DAMTrainer class allows for a seamless switch between on-the-fly and pre-computed logits. This flexibility is particularly beneficial for users with GPUs, enabling faster operations and more efficient testing experiments.

### Key Changes

#### New Parameters
- `generate_logits_on_fly`: A boolean parameter to control whether logits should be generated on-the-fly or pre-computed.
- `use_all_logits`: A boolean parameter to indicate if all logits should be used. This is only applicable when `generate_logits_on_fly` is True.

#### Assertions
- Added an assertion to ensure that `use_all_logits` cannot be True if `generate_logits_on_fly` is False.

#### Logits Computation
- When `generate_logits_on_fly` is True, logits for each individual model are computed dynamically during training.
- When `generate_logits_on_fly` is False, pre-computed logits are used, and the top-K logits are gathered using the provided indices.

#### Efficiency Improvements
- By allowing on-the-fly logits generation, users with GPUs can leverage their hardware to perform operations faster.
- This flexibility also aids in conducting various testing experiments more efficiently.

#### Code Updates
- Updated the `compute_loss` method to handle both on-the-fly and pre-computed logits.
- Modified the `compute_individual_logit_losses` method to accommodate the new parameters and logic.

### Benefits
- **Performance**: Users with GPUs can experience faster training times by generating logits on-the-fly.
- **Flexibility**: The ability to switch between on-the-fly and pre-computed logits provides greater flexibility for different use cases and testing scenarios.
- **Simplified Workflow**: When using on-the-fly logits generation, there is no need to manage and store top-K logits, simplifying the workflow.

### Usage
To use the new functionality, simply set the `generate_logits_on_fly` and `use_all_logits` parameters when initializing the DAMTrainer:

```python
trainer = DAMTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    lambda_coef=lambda_coef,
    lambda_coef_l1=lambda_coef_l1,
    lambda_coef_l2=lambda_coef_l2,
    temperature=temperature,
    use_kl=use_kl,
    use_mse=use_mse,
    use_entropy=use_entropy,
    base_model_path=base_model_name,
    generate_logits_on_fly=True,  # Enable on-the-fly logits generation
    use_all_logits=True,  # Use all logits when generating on-the-fly
)
```

## Help Us to Run More Experiments and Visualizations
Please help us to run more experiments on this concept and provide more visualizations. Your contributions and feedback are highly appreciated!
