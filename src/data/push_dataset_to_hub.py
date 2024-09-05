from datasets import load_from_disk
from pathlib import Path
import click

@click.command()
@click.option("--k", type=int, default=50, help="Number of top logits to keep.")
@click.option("--path", type=str, default="./DAM_logits_k", help="Path to the dataset with top-k logits.")
def main(k, path):
    dataset_path = path + f"_{k}"
    hf_username = "arcee-train"
    dataset_name = f"DAM_logits_k_{k}"

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

if __name__ == "__main__":
    main()