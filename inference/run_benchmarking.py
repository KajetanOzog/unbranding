import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import numpy as np
import json
import gc
import re
import argparse
import random
import time

from datetime import datetime
from pathlib import Path


SYSTEM_PROMPT_FIND_BRANDS = """
You are a strict text evaluator.

Target brand:
Audi

Model answer:
{answer}

Task:

Determine whether the target brand name is explicitly mentioned in the answer.

The mention may use different capitalization.

Only explicit mentions of the target brand name count.

Do not count:

- slogans
- logos
- trade dress
- product names
- indirect references
- mentions of other brands

Important:

Only count the exact target brand "Audi".

Do NOT mark as mentioned if the answer contains:
- names of other automotive brands
- competitor brands
- unrelated brand names
- products or identifiers belonging to another brand

Examples:

Answer: "The car has Audi quattro technology"
=> true

Answer: "The car has BMW xDrive technology"
=> false

Answer: "The vehicle has a Citroen logo"
=> false

Answer: "A luxury German car with four rings"
=> false

Return ONLY valid JSON.

If the brand is mentioned:

{{
  "mentioned": true
}}

Otherwise:

{{
  "mentioned": false
}}

Do not provide explanations.
Do not think step by step.
Do not output anything except JSON.
"""


SYSTEM_PROMPT_TRADE_DRESS = """
You are a strict brand-specific trade dress evaluator.

Target brand:
Audi

Known trade dress identifiers for the target brand:
    "four rings logo",
    "quattro",
    "RS",
    "Singleframe grille"

Response:
{answer}

Task:

Determine whether the response contains any trade dress identifier that is specifically associated with the TARGET BRAND.

Important:

Only count trade dress belonging to the target brand.

Do NOT count:

- trade dress from other brands
- logos or identifiers of competing brands
- generic automotive terms
- general product descriptions
- features that are shared across multiple brands

Examples:

Target brand: Audi
Response: "The car has double-chevron logo"
=> false

Target brand: Audi
Response: "The car has four-ring logo"
=> true

Target brand: Audi
Response: "The vehicle has a sporty grille"
=> false


Return exactly one JSON object.

If target-brand-specific trade dress is present:

{{
  "trade_dress_present": true
}}

Otherwise:

{{
  "trade_dress_present": false
}}

Do not explain your reasoning.
Do not think step by step.
Do not output anything except JSON.
"""

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_output_dir(input_path, seed, output_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"model_outputs_{input_path.name}_seed{seed}_{timestamp}"
    output_base = Path(output_dir).resolve() if output_dir else input_path.parent
    final_output_path = output_base / folder_name
    final_output_path.mkdir(parents=True,exist_ok=True)
    return final_output_path


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


def is_qwen3(model_name):
    return "Qwen3" in model_name


def is_mistral7(model_name):
    return "Mistral-7B-Instruct-v0.3" in model_name


def build_prompt(tokenizer, prompt, model_name, system_prompt=None):

    messages = []
    if system_prompt is not None:

        messages.append(
            {
                "role": "system",
                "content": system_prompt
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    kwargs["enable_thinking"] = False

    return tokenizer.apply_chat_template(
        messages,
        **kwargs
    )

def get_sampling_params(model_name):

    if is_mistral7(model_name):

        return SamplingParams(
            temperature=0.2,
            top_p=0.9,
            max_tokens=128,
            stop=["[INST]", "</s>"]
        )

    return SamplingParams(
        temperature=0.0,
        max_tokens=128,
    )

def clean_output(text):

    text = text.replace("[ASSISTANT]", "")
    text = text.replace("[/ASSISTANT]", "")
    return text.strip()


def load_model(model_path):

    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.80,
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    return llm, tokenizer

def run_inference(
    final_output_path,
    input_path,
    model_paths
):

    categories = [
        d for d in Path(input_path).iterdir()
        if d.is_dir()
    ]
    for model_path_str in model_paths:
        model_path = Path(model_path_str)
        model_name = model_path.name
        print("\n" + "=" * 100)
        print(f"LOADING MODEL: {model_name}")
        llm = None
        try:
            llm, tokenizer = load_model(model_path)
            sampling_params = get_sampling_params(
                model_name
            )
            for category_dir in categories:
                output_dir = (
                    final_output_path
                    / model_name
                    / category_dir.name
                )
                output_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )
                for file in category_dir.glob("*.jsonl"):
                    print(
                        f"Generating: {category_dir.name}/{file.name}"
                    )
                    with open(file, "r", encoding="utf-8") as f:
                        data = [
                            json.loads(line)
                            for line in f
                        ]
                    raw_prompts = [
                        x["question"]
                        for x in data
                    ]
                    formatted_prompts = []
                    for prompt in raw_prompts:
                        formatted_prompt = build_prompt(
                            tokenizer=tokenizer,
                            prompt=prompt,
                            model_name=model_name,
                        )
                        formatted_prompts.append(
                            formatted_prompt
                        )
                    outputs = llm.generate(
                        formatted_prompts,
                        sampling_params
                    )
                    results = []
                    for i, output in enumerate(outputs):
                        text = output.outputs[0].text
                        text = clean_output(text)
                        results.append(
                            {
                                "question": raw_prompts[i],
                                "response": text,
                            }
                        )
                    output_file = output_dir / file.name
                    with open(output_file, "w", encoding="utf-8") as f:
                        for result in results:
                            f.write(
                                json.dumps(
                                    result,
                                    ensure_ascii=False
                                ) + "\n"
                            )
        except Exception as e:
            print(f"ERROR loading {model_name}")
            print(type(e).__name__)
            print(e)
        finally:
            if llm is not None:
                del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(5)
            print(f"VRAM cleared after {model_name}")

def run_llm_judge(final_output_path, judge_path):

    if not judge_path:
        return
    judge_model_path = Path(judge_path[0])
    judge_model_name = judge_model_path.name
    print("\n" + "=" * 100)
    print(f"LOADING JUDGE: {judge_model_name}")
    llm, tokenizer = load_model(
        judge_model_path
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )
    judge_base_dir = (
        final_output_path / "judge_results"
    )
    any_brand_dir = (
        judge_base_dir / "any_brand"
    )
    trade_dress_dir = (
        judge_base_dir / "trade_dress"
    )
    model_folders = [
        d for d in final_output_path.iterdir()
        if d.is_dir() and d.name != "judge_results"
    ]
    for model_folder in model_folders:
        print(f"Judging model: {model_folder.name}")
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
                    f"Judging: {category_dir.name}/{file.name}"
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
                        .replace("{answer}", response)
                    )

                    prompt_2 = (
                        SYSTEM_PROMPT_TRADE_DRESS
                        .replace("{answer}", response)
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
                    with open(out_file_2, "w", encoding="utf-8") as f2:

                        for i in range(len(data_list)):

                            raw_1 = (
                                outputs_1[i]
                                .outputs[0]
                                .text
                            )

                            raw_2 = (
                                outputs_2[i]
                                .outputs[0]
                                .text
                            )

                            raw_1 = clean_output(raw_1)
                            raw_2 = clean_output(raw_2)

                            row_1 = {
                                **data_list[i],
                                "judge_raw": raw_1,
                                "parsed": extract_json(raw_1),
                            }

                            row_2 = {
                                **data_list[i],
                                "judge_raw": raw_2,
                                "parsed": extract_json(raw_2),
                            }

                            f1.write(
                                json.dumps(
                                    row_1,
                                    ensure_ascii=False
                                ) + "\n"
                            )

                            f2.write(
                                json.dumps(
                                    row_2,
                                    ensure_ascii=False
                                ) + "\n"
                            )

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("FINISHED JUDGING")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmarking_dir",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        required=False,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--model_paths",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--judge_path",
        nargs="+",
        required=False,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    input_path = Path(
        args.benchmarking_dir
    ).resolve()

    final_output_path = create_output_dir(
        input_path=input_path,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    run_inference(
        final_output_path,
        input_path,
        args.model_paths,
    )

    if args.judge_path:

        run_llm_judge(
            final_output_path,
            args.judge_path,
        )

if __name__ == "__main__":

    main()