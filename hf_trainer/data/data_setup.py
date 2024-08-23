from datasets import load_dataset

def setup_datasets_and_templates(tokenizer, dataset_names, example_count=None):
    """
    Setup datasets and apply templates dynamically based on provided dataset names and column structures.

    Args:
        tokenizer: The tokenizer with a chat template applied.
        dataset_names (list): A list of dataset names to load.
        example_count (int, optional): The number of examples to select from each dataset.

    Returns:
        list: A list of templated datasets.
    """

    # Define the chat template
    tokenizer.chat_template = "{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n{%- if message['role'] == 'system' -%}\n{%- set ns.found = true -%}\n{%- endif -%}\n{%- endfor -%}\n{%- if not ns.found -%}\n{{- '' + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' + '\\n\\n' -}}\n{%- endif %}\n{%- for message in messages %}\n{%- if message['role'] == 'system' -%}\n{{- '' + message['content'] + '\\n\\n' -}}\n{%- else -%}\n{%- if message['role'] == 'user' -%}\n{{-'### Instruction:\\n' + message['content'] + '\\n\\n'-}}\n{%- else -%}\n{{-'### Response:\\n' + message['content'] + '\\n\\n' -}}\n{%- endif -%}\n{%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n{{-'### Response:\\n'-}}\n{%- endif -%}"

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

    # Function to determine the user input and assistant output columns dynamically
    def determine_columns(dataset):
        possible_input_columns = ['instruction', 'question', 'query', 'text']
        possible_output_columns = ['output', 'answer', 'response']

        input_column = None
        output_column = None

        for col in dataset.column_names:
            if input_column is None and col in possible_input_columns:
                input_column = col
            if output_column is None and col in possible_output_columns:
                output_column = col

            if input_column and output_column:
                break

        if not input_column or not output_column:
            raise ValueError("Could not determine the appropriate input/output columns for the dataset.")

        return input_column, output_column

    # Load and process datasets
    templated_datasets = []
    for dataset_name in dataset_names:
        # Check if the dataset name includes a specific config/version
        if ':' in dataset_name:
            base_name, config_version = dataset_name.split(':')
            dataset = load_dataset(base_name, config_version)
        else:
            dataset = load_dataset(dataset_name)

        # Select examples if example_count is provided
        if example_count:
            dataset = select_examples(dataset, example_count)

        # Determine the input and output columns
        input_column, output_column = determine_columns(dataset)

        # Apply templates
        templated_dataset = dataset.map(lambda row: {'text': tokenizer.apply_chat_template(
            [{'role': 'user', 'content': row[input_column]}, {'role': 'assistant', 'content': row[output_column]}], 
            tokenize=False).strip()})

        templated_datasets.append(templated_dataset)

    return templated_datasets
