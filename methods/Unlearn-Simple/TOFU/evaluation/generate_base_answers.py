from pathlib import Path
import gc

import torch

from vllm import (
    LLM,
    SamplingParams,
)

from transformers import AutoTokenizer

from utils.config import (
    load_configs,
    get_brand_name,
    get_base_model_path,
    get_model_family,
    get_generation_config,
    get_results_config,
)

from utils.datasets import (
    load_jsonl,
    save_jsonl,
    get_evaluation_datasets,
    get_questions,
    build_prompts,
)

from utils.records import (
    build_output_record,
)

from utils.results import (
    get_base_output_dir,
)


def load_model(
    model_path,
    seed,
):

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        seed=seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    return (
        llm,
        tokenizer,
    )


def unload_model(llm):

    del llm

    gc.collect()

    torch.cuda.empty_cache()


def generate_answers_for_dataset(
    llm,
    tokenizer,
    dataset,
    max_new_tokens,
    seed,
):

    questions = get_questions(
        dataset
    )

    prompts = build_prompts(
        questions,
        tokenizer,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        seed=seed,
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
    )

    return [
        output.outputs[0].text.strip()
        for output in outputs
    ]


def main():

    experiment_cfg, brand_cfg = (
        load_configs()
    )

    brand_name = get_brand_name(
        experiment_cfg
    )

    model_family = get_model_family(
        experiment_cfg
    )

    base_model_path = (
        get_base_model_path(
            experiment_cfg
        )
    )

    generation_cfg = (
        get_generation_config(
            experiment_cfg
        )
    )

    results_cfg = (
        get_results_config(
            experiment_cfg
        )
    )

    seed = experiment_cfg[
        "evaluation"
    ]["seed"]

    datasets = (
        get_evaluation_datasets(
            experiment_cfg,
            brand_cfg,
        )
    )

    output_dir = (
        get_base_output_dir(
            answers_dir=results_cfg[
                "answers_dir"
            ],
            brand_name=brand_name,
            model_family=model_family,
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("BASE MODEL")
    print(base_model_path)
    print("=" * 80)

    llm, tokenizer = load_model(
        model_path=base_model_path,
        seed=seed,
    )

    for dataset_name, dataset_path in datasets.items():

        print()
        print(
            f"Dataset: {dataset_name}"
        )

        dataset = load_jsonl(
            dataset_path
        )

        model_answers = generate_answers_for_dataset(
            llm=llm,
            tokenizer=tokenizer,
            dataset=dataset,
            max_new_tokens=generation_cfg[
                "max_new_tokens"
            ],
            seed=seed,
        )

        records = []

        for sample, answer in zip(
            dataset,
            model_answers,
        ):

            records.append(
                build_output_record(
                    sample,
                    answer,
                )
            )

        save_jsonl(
            records,
            output_dir
            / f"{dataset_name}.jsonl",
        )

    unload_model(llm)


if __name__ == "__main__":
    main()