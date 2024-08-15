# Differentiable Adaptive Merging (DAM) for Large Language Models

This repository contains the implementation of the Differentiable Adaptive Merging (DAM) technique, a novel method for efficiently merging large language models (LLMs) with varying capabilities. This approach optimizes the integration process by learning the optimal balance between models, reducing computational costs, and improving the quality of the merged models.

## Project Structure

- `dam.py`: Contains the `DAMLayer` class, which implements the differentiable adaptive merging logic.
- `utils.py`: Utility functions used across the project, including functions for finding linear layers in the models.
- `train.py`: The main script that sets up the models, applies the DAM technique, and handles training and evaluation.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Set Up a Virtual Environment (Optional)

It is recommended to use a virtual environment to manage dependencies:

```bash
python3 -m venv dam-env
source dam-env/bin/activate  # On Windows use `dam-env\Scripts\activate`
```

### 3. Install the Required Packages

You can install the required Python packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model using the DAM technique, simply run the `train.py` script:

```bash
python train.py
```

This will:
- Load the pre-trained models specified in `train.py`.
- Apply the DAM technique to merge the models.
- Train the merged model and save it to the specified directory.