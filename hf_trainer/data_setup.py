from datasets import load_dataset

def setup_datasets_and_templates(tokenizer, example_count=None):
    # Define the chat template
    tokenizer.chat_template = "{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n{%- if message['role'] == 'system' -%}\n{%- set ns.found = true -%}\n{%- endif -%}\n{%- endfor -%}\n{%- if not ns.found -%}\n{{- '' + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' + '\\n\\n' -}}\n{%- endif %}\n{%- for message in messages %}\n{%- if message['role'] == 'system' -%}\n{{- '' + message['content'] + '\\n\\n' -}}\n{%- else -%}\n{%- if message['role'] == 'user' -%}\n{{-'### Instruction:\\n' + message['content'] + '\\n\\n'-}}\n{%- else -%}\n{{-'### Response:\\n' + message['content'] + '\\n\\n' -}}\n{%- endif -%}\n{%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n{{-'### Response:\\n'-}}\n{%- endif -%}"

    # Load datasets
    dataset_A = load_dataset("p1atdev/ichikara-instruction", '20231115-1').rename_column("text", "instruction")
    dataset_B = load_dataset("microsoft/orca-math-word-problems-200k")
    dataset_C = load_dataset("meta-math/MetaMathQA")

    # Function to select a specific number of examples
    def select_examples(dataset, example_count):
        if "train" in dataset:
            split = "train"
        else:
            split = list(dataset.keys())[0]  # Use the first available split if "train" is not present
        try:
            return dataset[split].select(range(example_count))
        except Exception as e:
            print(f"Error selecting examples from {split} split: {e}")
            return dataset[split]  # Return the whole split if there's an issue

    # Select examples if example_count is provided
    if example_count:
        dataset_A = select_examples(dataset_A, example_count)
        dataset_B = select_examples(dataset_B, example_count)
        dataset_C = select_examples(dataset_C, example_count)

    # Apply templates to datasets
    templated_dataset_A = dataset_A.map(lambda row: {'text': tokenizer.apply_chat_template(
        [{'role': 'user', 'content': row['instruction']}, {'role': 'assistant', 'content': row['output']}], 
        tokenize=False).strip()})
    
    templated_dataset_B = dataset_B.map(lambda row: {'text': tokenizer.apply_chat_template(
        [{'role': 'user', 'content': row['question']}, {'role': 'assistant', 'content': row['answer']}], 
        tokenize=False).strip()})
    
    templated_dataset_C = dataset_C.map(lambda row: {'text': tokenizer.apply_chat_template(
        [{'role': 'user', 'content': row['query']}, {'role': 'assistant', 'content': row['response']}], 
        tokenize=False).strip()})

    # Return the list of templated datasets
    return [templated_dataset_A, templated_dataset_B, templated_dataset_C]
