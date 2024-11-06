"""
JBLiMP: Japanese Benchmark of Linguistic Minimal Pairs
https://aclanthology.org/2023.findings-eacl.117/

JBLiMP is a novel dataset for targeted syntactic evaluations of language models in Japanese. JBLiMP consists of 331 minimal pairs, which are created based on acceptability judgments extracted from journal articles in theoretical linguistics. These minimal pairs are grouped into 11 categories, each covering a different linguistic phenomenon.

Homepage: https://github.com/osekilab/JBLiMP/tree/main
"""
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
from lm_eval.tasks.blimp import BlimpTask

_CITATION = """
@inproceedings{Someya2023JBLiMPJB,
  title={JBLiMP: Japanese Benchmark of Linguistic Minimal Pairs},
  author={Taiga Someya and Yohei Oseki},
  booktitle={Findings},
  year={2023}
}
"""  # noqa: W605


class JBlimpTask(BlimpTask):
    VERSION = 0
    DATASET_PATH = "polm-stability/jblimp"
    DATASET_NAME = None


class JBlimp(JBlimpTask):
    DATASET_NAME = "jblimp"

    # NOTE: This is very confusing, but while BLiMP uses keys like `sentence_good`,
    # JBLiMP uses keys like `good_sentence`.

    def doc_to_decontamination_query(self, doc):
        return doc["good_sentence"] + " " + doc["bad_sentence"]

    def construct_requests(self, doc, ctx):
        assert not ctx

        # Calculate the loglikelihood for the good and the bad sentence.
        # Note that loglikelihood translates the "" prefix to the "<|endoftext|>" token
        return [
            rf.loglikelihood("", doc["good_sentence"]),
            rf.loglikelihood("", doc["bad_sentence"]),
        ]
