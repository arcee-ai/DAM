#!/usr/bin/env python
# This script runs eval in the cluster. Use it as a basis for your own harnesses.
from run_eval import build_executor, run_job
from run_eval import JAEVAL8_TASKS, JAEVAL8_FEWSHOT
from main_eval import main as main_eval


def build_task_list(tasks, prompt):
    out = []
    # Some tasks don't have a prompt version
    promptless = ["xwinograd_ja"]
    for task in tasks:
        if task not in promptless:
            out.append(f"{task}-{prompt}")
        else:
            out.append(task)
    return out


def main():
    executor = build_executor("eval", gpus_per_task=8, cpus_per_gpu=12)

    tasks = build_task_list(JAEVAL8_TASKS, "0.3")
    eval_args = {
        "tasks": tasks,
        "num_fewshot": JAEVAL8_FEWSHOT,
        "model": "hf-causal",
        "model_args": "pretrained=rinna/japanese-gpt-1b,use_fast=False",
        "device": "cuda",
        "limit": 100,
        "verbose": True,
    }

    run_job(executor, main_eval, eval_args=eval_args, output_path="./check.json")


if __name__ == "__main__":
    main()
