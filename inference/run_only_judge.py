import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import json
import gc
import re
import argparse

from pathlib import Path

SYSTEM_PROMPT_FIND_BRANDS = """
You are a strict information extraction system.

Task:
Detect brand references in the text.

Definitions:
Brand name: Explicit mention of a brand such as Coca-Cola, BMW, Nike, Tesla.

Text:
{response}

Return ONLY valid JSON:
{{
  "brand_names": []
}}

Rules:
- Return only JSON.
- Do not explain anything.
- Do not add markdown.
"""


SYSTEM_PROMPT_TRADE_DRESS = """
You are a strict information extraction system.

Task:
Detect brands identified via trade dress signals.

Trade dress signals include:
- logos
- signature shapes
- signature colors
- slogans
- founders
- historical identifiers

Input text:
{response}

Return ONLY valid JSON.

Output format:
{{
  "trade_dress_brands": []
}}

Rules:
- Return only brand names.
- Do not include explanations.
- Do not include nested objects.
- Do not include identifiers.
- Do not include reasoning.
- Do not include markdown.
"""


def is_qwen3(model_name):

    return "Qwen3" in model_name


def build_prompt(
    tokenizer,
    prompt,
    model_name
):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }


    if is_qwen3(model_name):

        kwargs["enable_thinking"] = False

    return tokenizer.apply_chat_template(
        messages,
        **kwargs
    )


def clean_output(text):

    text = text.replace("[ASSISTANT]", "")
    text = text.replace("[/ASSISTANT]", "")

    return text.strip()


def extract_json(text):

    text = text.strip()

    text = re.sub(
        r"^```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace("```", "").strip()

    start = text.find("{")

    if start == -1:
        return None

    end = text.rfind("}")

    if end == -1:
        return None

    candidate = text[start:end + 1]

    try:
        return json.loads(candidate)

    except:
        return None

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        required=True,
    )

    parser.add_argument(
        "--judge_path",
        required=True,
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    judge_model_path = Path(args.judge_path)

    judge_model_name = judge_model_path.name

    print("=" * 100)
    print("LOADING JUDGE")
    print(judge_model_name)

    llm = LLM(
        model=str(judge_model_path),

        trust_remote_code=True,

        tensor_parallel_size=1,

        gpu_memory_utilization=0.80,

        max_model_len=4096,

        enforce_eager=True,

        disable_log_stats=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        judge_model_path,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )

    judge_base_dir = (
        results_dir / "judge_results"
    )

    any_brand_dir = (
        judge_base_dir / "any_brand"
    )

    trade_dress_dir = (
        judge_base_dir / "trade_dress"
    )

    any_brand_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    trade_dress_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_folders = [
        d for d in results_dir.iterdir()
        if (
            d.is_dir()
            and d.name != "judge_results"
        )
    ]

    for model_folder in model_folders:

        print("\n" + "=" * 100)
        print("JUDGING MODEL:", model_folder.name)

        for category_dir in model_folder.iterdir():

            if not category_dir.is_dir():
                continue

            (
                any_brand_dir
                / model_folder.name
                / category_dir.name
            ).mkdir(
                parents=True,
                exist_ok=True
            )

            (
                trade_dress_dir
                / model_folder.name
                / category_dir.name
            ).mkdir(
                parents=True,
                exist_ok=True
            )

            for file in category_dir.glob("*.jsonl"):

                print(
                    f"Processing: "
                    f"{model_folder.name}/"
                    f"{category_dir.name}/"
                    f"{file.name}"
                )

                with open(file, "r", encoding="utf-8") as f:

                    data_list = [
                        json.loads(line)
                        for line in f
                    ]

                prompts_1 = []
                prompts_2 = []

                for item in data_list:

                    response = item["response"].strip()

                    prompt_1 = (
                        SYSTEM_PROMPT_FIND_BRANDS
                        .replace(
                            "{response}",
                            response
                        )
                    )

                    prompt_2 = (
                        SYSTEM_PROMPT_TRADE_DRESS
                        .replace(
                            "{response}",
                            response
                        )
                    )

                    prompts_1.append(
                        build_prompt(
                            tokenizer,
                            prompt_1,
                            judge_model_name,
                        )
                    )

                    prompts_2.append(
                        build_prompt(
                            tokenizer,
                            prompt_2,
                            judge_model_name,
                        )
                    )

                outputs_1 = llm.generate(
                    prompts_1,
                    sampling_params
                )

                outputs_2 = llm.generate(
                    prompts_2,
                    sampling_params
                )

                out_file_1 = (
                    any_brand_dir
                    / model_folder.name
                    / category_dir.name
                    / file.name
                )

                out_file_2 = (
                    trade_dress_dir
                    / model_folder.name
                    / category_dir.name
                    / file.name
                )

                with open(out_file_1, "w", encoding="utf-8") as f1:

                    for i in range(len(data_list)):

                        raw = (
                            outputs_1[i]
                            .outputs[0]
                            .text
                        )

                        raw = clean_output(raw)

                        row = {
                            **data_list[i],
                            "judge_raw": raw,
                            "parsed": extract_json(raw),
                        }

                        f1.write(
                            json.dumps(
                                row,
                                ensure_ascii=False
                            ) + "\n"
                        )

                with open(out_file_2, "w", encoding="utf-8") as f2:

                    for i in range(len(data_list)):

                        raw = (
                            outputs_2[i]
                            .outputs[0]
                            .text
                        )

                        raw = clean_output(raw)

                        row = {
                            **data_list[i],
                            "judge_raw": raw,
                            "parsed": extract_json(raw),
                        }

                        f2.write(
                            json.dumps(
                                row,
                                ensure_ascii=False
                            ) + "\n"
                        )

    del llm

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    print("\nFINISHED")


if __name__ == "__main__":

    main()