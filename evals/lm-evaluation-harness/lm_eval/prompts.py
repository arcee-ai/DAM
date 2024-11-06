def jslm_beta(task):
    """JSLM Beta uses a different prompt for JCommonSenseQA."""
    if task == "jcommonsenseqa":
        return "0.2.1"
    else:
        return "0.2"


PROMPT_CODES = {
    "user": "0.0",
    "jgpt": "0.1",
    "fintan": "0.2",
    "fintan2": "0.2.1",
    "ja-alpaca": "0.3",
    "rinna-sft": "0.4",
    "rinna-bilingual": "0.5",
    "llama2": "0.6",
    "jslm-beta": jslm_beta,
}


def get_prompt_code(short_name, task=None):
    """Get the prompt code given a short name.

    Usually, this is a simple dictionary lookup. But it can depend on the task
    sometimes.
    """
    code = PROMPT_CODES[short_name]

    if callable(code):
        return callable(task)
    else:
        return code
