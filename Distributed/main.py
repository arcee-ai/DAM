import os

def main():
    # Step 1: Prepare datasets with logits
    os.system("python dataset_preparation/prepare_datasets.py")

    # Step 2: Merge models
    os.system("python model_merging/merge_models.py")

    # Step 3: Train the merged model
    os.system("python training/train_model.py")

if __name__ == "__main__":
    main()
