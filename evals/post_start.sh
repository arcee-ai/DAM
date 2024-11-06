#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value

git config --global credential.helper store
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential
git clone https://huggingface.co/datasets/arcee-ai/DAM-evals /evals
cd /evals; bash runpod.sh