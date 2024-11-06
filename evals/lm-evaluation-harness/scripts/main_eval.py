import os
import argparse
import json
import logging
import fnmatch

from lm_eval import tasks, evaluator

logging.getLogger("openai").setLevel(logging.WARNING)


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--num_fewshot", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=str, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    # TODO This is deprecated and throws an error, remove it
    parser.add_argument("--provide_description", action="store_true")

    return parser.parse_args()


def clean_args(args) -> dict:
    """Handle conversion to lists etc. for args"""

    assert not args.provide_description, "provide-description is not implemented"

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        args.tasks = tasks.ALL_TASKS
    else:
        args.tasks = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {args.tasks}")
    if args.num_fewshot is not None:
        args.num_fewshot = [int(n) for n in args.num_fewshot.split(",")]

    if args.limit is not None:
        args.limit = [
            int(n) if n.isdigit() else float(n) for n in args.limit.split(",")
        ]

    return vars(args)


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = []
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.append(matching)
    return task_names


def main(eval_args: dict, description_dict_path: str = None, output_path: str = None):
    """Run evaluation and optionally save output.

    For a description of eval args, see `simple_evaluate`.
    """
    if description_dict_path:
        with open(description_dict_path, "r") as f:
            eval_args["description_dict"] = json.load(f)

    results = evaluator.simple_evaluate(**eval_args)

    dumped = json.dumps(results, indent=2, ensure_ascii=False)
    print(dumped)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(dumped)

    return results


if __name__ == "__main__":
    args = parse_args()
    args = clean_args(args)

    # This is not used
    args.pop("provide_description", None)
    # treat non-eval args separately
    description_dict_path = args.get("description_dict_path", None)
    args.pop("description_dict_path", None)
    output_path = args.get("output_path", None)
    args.pop("output_path", None)

    results = main(args, description_dict_path, output_path)

    print(
        f"{args['model']} ({args['model_args']}), limit: {args['limit']}, "
        f"num_fewshot: {args['num_fewshot']}, batch_size: {args['batch_size']}"
    )
    print(evaluator.make_table(results))
