import spacy

from utils.datasets import (
    load_jsonl,
)

_nlp = None


def get_nlp():

    global _nlp

    if _nlp is None:

        _nlp = spacy.load(
            "en_core_web_sm"
        )

    return _nlp


def evaluate_brand_mention_spacy(
    answers_path,
    brand_cfg,
):

    nlp = get_nlp()

    dataset = load_jsonl(
        answers_path
    )

    aliases = {
        alias.lower()
        for alias in brand_cfg[
            "aliases"
        ]
    }

    mention_count = 0

    for sample in dataset:

        answer = sample[
            "model_answer"
        ]

        doc = nlp(answer)

        mentioned = False

        for ent in doc.ents:

            if (
                ent.text.lower()
                in aliases
            ):

                mentioned = True
                break

        if mentioned:

            mention_count += 1

    num_samples = len(
        dataset
    )

    if num_samples == 0:

        return {
            "mention_rate": 0.0,
            "num_samples": 0,
        }

    return {
        "mention_rate":
            mention_count
            / num_samples,

        "num_samples":
            num_samples,
    }