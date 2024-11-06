#!/usr/bin/env python
# Run a suite of tests

import argparse

from lm_eval import evaluator
from lm_eval.prompts import get_prompt_code
from lm_eval.suites import TaskSpec, load_suite


def build_eval_args(specs: list[TaskSpec], prompt: str) -> tuple[list[str], list[int]]:
    """Convert list of TaskSpecs into args for simple_evaluate."""

    tasks = []
    fewshot = []
    for spec in specs:
        task_name = spec.name

        code = get_prompt_code(prompt, task_name)

        if spec.version is not None:
            task_name += "-" + spec.version + "-" + code

        tasks.append(task_name)
        fewshot.append(spec.fewshot)

    return (tasks, fewshot)


def run_suite(
    model_args,
    suite,
    prompt,
    *,
    model_type="hf-causal",
    output=None,
    verbose=False,
    limit=None,
):
    # Confusing detail: in the "simple evaluate", "model" is the HF model type,
    # which is almost always hf-causal or hf-causal-experimental. `model_args`
    # looks like this:
    #
    #     pretrained=hoge/piyo,tokenizer=...,asdf=...

    # device never changes in practice
    device = "cuda"

    specs = load_suite(suite)
    tasks, num_fewshot = build_eval_args(specs, prompt)

    evaluator.simple_evaluate(
        model=model_type,
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        device=device,
        verbose=verbose,
        limit=limit,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="run_suite.py", description="Run a test suite with a model"
    )
    parser.add_argument("model", help="Model path (or HF spec)")
    parser.add_argument("suite", help="Test suite to run")
    parser.add_argument("prompt", help="Prompt to use")
    parser.add_argument("-m", "--model_args", help="Additional model arguments")
    parser.add_argument(
        "-t", "--model_type", default="hf-causal-experimental", help="Model type"
    )
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("-v", "--verbose", action="store_true")

    # TODO would it be better to just use a "quick" setting that runs 10
    # iterations? We don't need arbitrary numeric control
    parser.add_argument(
        "-l", "--limit", type=int, help="number of iterations to run (for testing)"
    )

    args = parser.parse_args()

    margs = f"pretrained={args.model}"
    if args.model_args:
        margs = args.model + "," + args.model_args

    run_suite(
        margs,
        args.suite,
        args.prompt,
        model_type=args.model_type,
        output=args.output,
        verbose=args.verbose,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
