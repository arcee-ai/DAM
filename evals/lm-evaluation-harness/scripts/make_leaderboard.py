import os
from pathlib import Path
import json
import pandas as pd
import blobfile as bf


OTHERS = {
    "pretrained=abeja/gpt-neox-japanese-2.7b": "abeja-gpt-neox-japanese-2.7b",
}
RINNA = {
    "pretrained=rinna/japanese-gpt-1b,use_fast=False": "rinna-japanese-gpt-1b",
    "pretrained=rinna/japanese-gpt-neox-3.6b,use_fast=False": "rinna-japanese-gpt-neox-3.6b",
    "pretrained=rinna/japanese-gpt-neox-3.6b-instruction-ppo,use_fast=False": "rinna-japanese-gpt-neox-3.6b-instruction-ppo",
    "pretrained=rinna/japanese-gpt-neox-3.6b-instruction-sft,use_fast=False": "rinna-japanese-gpt-neox-3.6b-instruction-sft",
    "pretrained=rinna/japanese-gpt-neox-3.6b-instruction-sft-v2,use_fast=False": "rinna-japanese-gpt-neox-3.6b-instruction-sft-v2",
}
CYBERAGENT = {
    "pretrained=cyberagent/open-calm-medium": "cyberagent-open-calm-medium",
    "pretrained=cyberagent/open-calm-large": "cyberagent-open-calm-large",
    "pretrained=cyberagent/open-calm-1b": "cyberagent-open-calm-1b",
    "pretrained=cyberagent/open-calm-3b": "cyberagent-open-calm-3b",
    "pretrained=cyberagent/open-calm-7b": "cyberagent-open-calm-7b",
}
MODELARGS2ID = {**OTHERS, **RINNA, **CYBERAGENT}

TASK2MAINMETRIC = {
    "jcommonsenseqa": "acc",
    "jnli": "acc",
    "marc_ja": "acc",
    "jsquad": "exact_match",
    "jaquad": "exact_match",
    "xlsum_ja": "rouge2",
}
TASK2SHOT = {
    "jcommonsenseqa": 2,
    "jnli": 3,
    "marc_ja": 3,
    "jsquad": 3,
    "jaquad": 3,
    "xlsum_ja": 1,
}


def get_class(model_args):
    if model_args in RINNA:
        return "rinna"
    elif model_args in CYBERAGENT:
        return "cyberagent"
    elif model_args in OTHERS:
        return ""
    else:
        raise NotImplementedError


def get_score(metric: str, value):
    if metric == "acc":
        return value * 100
    return value


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


def model_hyperlink(link, model_name):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'


models_dir = "models"
url_repo = "https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/{}"
url_hf = "https://huggingface.co/{org}/{model_name}"

files = _list_json_files_recursively(models_dir)

res_dict = {}


def add_data(key, value):
    global res_dict
    if key not in res_dict:
        res_dict[key] = [value]
    else:
        res_dict[key].append(value)


for file in files:
    if "experiments" in file or "community" in file:
        continue
    with open(file) as f:
        info = json.load(f)
    results = info["results"]
    # only 8 tasks
    if len(results) != 8:
        continue
    model_id = os.path.basename(os.path.dirname(file))
    org = model_id.split("-")[0]
    model_name = model_id[len(org) + 1 :]
    add_data(
        "model",
        model_hyperlink(url_hf.format(org=org, model_name=model_name), model_id),
    )
    p = os.path.join(os.path.dirname(file), "harness.sh")
    add_data("eval script", model_hyperlink(url_repo.format(p), p))
    scores = []
    for k, v in results.items():
        if "acc" in v:
            # to percent
            score = v["acc"] * 100
        elif "exact_match" in v:
            score = v["exact_match"]
        elif "rouge2" in v:
            score = v["rouge2"]
        else:
            NotImplementedError(v.keys())
        k = k.split("-")[0]
        add_data(k, round(score, 2))
        scores.append(score)
    add_data("average", round(sum(scores) / (len(scores) * 100) * 100, 2))
df = pd.DataFrame.from_dict(res_dict)
df = df[
    [
        "model",
        "average",
        "jcommonsenseqa",
        "jnli",
        "marc_ja",
        "jsquad",
        "jaqket_v2",
        "xlsum_ja",
        "xwinograd_ja",
        "mgsm",
        "eval script",
    ]
]
df.sort_values(by=["average"], inplace=True, ascending=False)
df.to_csv("jp_llm_leaderboard.csv", index=False)
df.to_markdown("jp_llm_leaderboard.md", index=False)
