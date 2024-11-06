# Functionality related to "eval suites". A suite is a collection of tasks with
# options pre-configured. Different models can be run with the same suite to
# compare them.
import configparser
from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

# This file is the path where suite configs go
SUITE_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / "configs"


@dataclass
class TaskSpec:
    """Specification of a task in an eval suite.

    A suite is a list of these specs, plus a prompt."""

    # The real arguments have to be massaged into messy strings and parallel
    # lists, but this is a more reasonable structure - we can handle conversion
    # separately.

    name: str
    fewshot: int
    version: Optional[str]


def load_suite(name):
    """Read in configuration for a test suite.

    A suite will have a config file named something like `my_suite.conf`. For
    each task in the file, a version, fewshot config, and any other details
    will be specified.

    Example entry:

        [tasks.mgsm]
        version = 1.0
        fewshot = 5
    """
    conf = configparser.ConfigParser()
    conf.read(SUITE_DIR / (name + ".conf"))

    specs = []
    for key, val in conf.items():
        if not key.startswith("tasks."):
            continue

        spec = TaskSpec(
            name=key.split(".", 1)[1],
            version=val.get("version", None),
            fewshot=int(val["fewshot"]),
        )
        specs.append(spec)
    return specs
