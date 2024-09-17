from datasets import load_dataset
from huggingface_hub import HfApi


# Load the dataset
dataset = load_dataset("kaxap/pg-wikiSQL-sql-instructions-80k")

# Define a function to combine columns
def combine_columns(example):
    return {
        "text": f"{example['question']} : {example['create_table_statement']}",
        "sql_query": example["sql_query"],
        "wiki_sql_table_id": example["wiki_sql_table_id"]
    }

# Apply the transformation to all splits
transformed_dataset = dataset.map(combine_columns)

# Keep only the desired columns
transformed_dataset = transformed_dataset.remove_columns(["question", "create_table_statement"])

# Save the transformed dataset
# Push the transformed dataset to the Hugging Face Hub as a private dataset
api = HfApi()
transformed_dataset.push_to_hub(
    "arcee-train/transformed_pg_wikiSQL_dataset",
    private=True
)

print("Transformed dataset pushed to Hugging Face Hub successfully.")