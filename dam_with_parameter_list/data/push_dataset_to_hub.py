from datasets import load_from_disk
from pathlib import Path

dataset_path = "./dataset_with_logits"
hf_username = "your-username"
dataset_name = "your-dataset-name"

dataset_path = Path(dataset_path)

if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

try:
    # Load the dataset from the saved directory
    dataset = load_from_disk(dataset_path)
except Exception as e:
    raise Exception(f"Error loading dataset from {dataset_path}: {e}")

# Push the dataset to Hugging Face Hub
dataset.push_to_hub(f"{hf_username}/{dataset_name}")

print("Dataset successfully uploaded to Hugging Face!")