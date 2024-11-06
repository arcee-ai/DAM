import submitit
from submitit.helpers import CommandFunction
import argparse
from notify import notify
from pathlib import Path

from main_eval import main as main_eval

# These are the standard 8 tasks
JAEVAL8_TASKS = [
    "jcommonsenseqa-1.1",
    "jnli-1.1",
    "marc_ja-1.1",
    "jsquad-1.1",
    "jaqket_v2-0.2",
    "xlsum_ja-1.0",
    "xwinograd_ja",
    "mgsm-1.0",
]
JAEVAL8_FEWSHOT = [3, 3, 3, 2, 1, 1, 0, 5]


def eval_task():
    args = {
        "tasks": ["jsquad-1.1-0.2"],
        "num_fewshot": [1],
        "model": "hf-causal",
        "model_args": "pretrained=rinna/japanese-gpt-1b,use_fast=False",
        "device": "cuda",
        "limit": 100,
        "verbose": True,
    }

    main_eval(args, output_path="./check.json")


def build_executor(
    name: str,
    gpus_per_task: int,
    cpus_per_gpu: int,
    timeout: int = 0,
    partition: str = "g40",
    account: str = "stablegpt",
):
    base_args = {
        # just "gpus" does not work
        "slurm_gpus_per_task": 8,
        "name": "eval",
        "slurm_account": "stablegpt",
        "slurm_partition": "g40",
        "slurm_cpus_per_gpu": 12,
        # Default timeout is 5 minutes???
        "timeout_min": 0,
    }

    executor = submitit.AutoExecutor(folder="./logs")
    executor.update_parameters(**base_args)
    return executor


def build_shell_script_task(script_path, args, repo):
    """This is how you can wrap an existing harness.sh.

    This is not currently used.
    """
    task = CommandFunction(["bash", args.harness_script], cwd=repo)
    return task


def run_job(executor, task, *args, **kwargs):
    """Given an executor and a task, run the task with error reporting.

    `executor` should be a submitit executor.

    `task` can be a CommandFunction from submitit, which wraps a shell script,
    or a Python function. Further positional or keyword arguments are passed to
    the function.
    """
    job = executor.submit(task, *args, **kwargs)
    print("Submitted job")
    print("See log at:")
    print(f"\t{job.paths.stdout}")

    try:
        output = job.result()
        print("Job finished successfully!")
        notify(f":white_check_mark: Eval Finished for `{job}`")
        return output
    except Exception as ee:  # noqa: F841
        # submitit doesn't seem to have a parent class for their exceptions, so
        # just catch everything. We want to be aware of any failure anyway.
        # If this is noisy we can ignore certain kinds of early failures.

        msg = f"""
            :rotating_light: Eval failed for `{job}`

            See `{job.paths.stderr}`
            """.strip()
        notify(msg)
        raise


def run_eval_shell_script():
    parser = argparse.ArgumentParser(
        prog="run-eval",
        description="Run eval harness",
    )

    parser.add_argument("harness_script")

    args = parser.parse_args()

    base_args = {
        # just "gpus" does not work
        "slurm_gpus_per_task": 8,
        "name": "eval",
        "slurm_account": "stablegpt",
        "slurm_partition": "g40",
        "slurm_cpus_per_gpu": 12,
        # Default timeout is 5 minutes???
        "timeout_min": 0,
    }

    executor = submitit.AutoExecutor(folder="./logs")
    executor.update_parameters(**base_args)

    # Harness scripts expect the cwd to be the repo root
    spath = Path(args.harness_script)
    repo = str(spath.parent.parent.parent.parent)
    print("repo path:", repo)
    # the eval harness relies on validating cli args, so it's difficult to run
    # directly from Python. Use the harness.sh scripts for now.
    # Also note this needs to be a list of strings.
    harness = CommandFunction(["bash", args.harness_script], cwd=repo)

    job = executor.submit(harness)
    print("Submitted job")
    print("See log at:")
    print(f"\t{job.paths.stdout}")

    try:
        output = job.result()
        print("Job finished successfully!")
        notify(f":white_check_mark: Eval Finished for `{args.harness_script}`")
        return output
    except Exception as ee:  # noqa: F841
        # submitit doesn't seem to have a parent class for their exceptions, so
        # just catch everything. We want to be aware of any failure anyway.
        # If this is noisy we can ignore certain kinds of early failures.

        msg = f"""
            :rotating_light: Eval failed for `{args.harness_script}`

            See `{job.paths.stderr}`
            """.strip()
        notify(msg)
        raise


def run_eval_submitit():
    """Run evaluation using submitit."""
    executor = build_executor()
    # By wrapping everything in a function, we don't have to pass args.
    run_job(executor, eval_task)


if __name__ == "__main__":
    run_eval_shell_script()
