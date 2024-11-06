import os
import json
import argparse
import pandas as pd
import blobfile as bf


def _list_json_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["json"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_json_files_recursively(full_path))
    return results


MODE2TASKS = {"jglue": ["jcommonsenseqa", "jnli", "marc_ja", "jsquad"]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", required=True, type=str, help="could be .json or dir"
    )
    parser.add_argument("--mode", default="all", choices=["jglue", "all"])
    args = parser.parse_args()
    if os.path.isfile(args.output_path):
        files = [args.output_path]
    else:
        files = _list_json_files_recursively(args.output_path)

    for file in files:
        with open(file) as f:
            info = json.load(f)
        results = info["results"]
        data = {"model_name": os.path.basename(file).replace(".json", "")}
        scores = []
        for k, v in results.items():
            if args.mode != "all":
                task_name = k.split("-")[0]
                if task_name not in MODE2TASKS[args.mode]:
                    continue
            if "acc" in v:
                # to percent
                score = v["acc"] * 100
            elif "exact_match" in v:
                score = v["exact_match"]
            elif "rouge2" in v:
                score = v["rouge2"]
            else:
                NotImplementedError(v.keys())
            data[k] = round(score, 2)
            scores.append(score)
        data["average"] = round(sum(scores) / (len(scores) * 100) * 100, 2)
        dumped = json.dumps(data, indent=2)
        print(dumped)
