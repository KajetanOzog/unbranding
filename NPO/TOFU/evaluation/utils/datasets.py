import json
from pathlib import Path


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

def load_json(path):
    with open(path, "r", encoding="utf-8",) as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_evaluation_datasets(experiment_cfg, brand_cfg):

    datasets = {}
    datasets.update(brand_cfg["datasets"])
    datasets.update(experiment_cfg["datasets"])

    return datasets

def get_questions(dataset):

    return [
        sample["question"]
        for sample in dataset
    ]

def build_prompts(
    questions,
    tokenizer,
):

    messages_list = [
        [
            {
                "role": "user",
                "content": question,
            }
        ]
        for question in questions
    ]

    return [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for messages in messages_list
    ]