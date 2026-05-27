import torch

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM
)

import numpy as np
import json
import gc
import argparse
import random
import time

from datetime import datetime
from pathlib import Path


# =========================================================
# HELPERS
# =========================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_output_dir(
    input_path,
    seed,
    output_dir
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    folder_name = (
        f"model_outputs_{input_path.name}_"
        f"seed{seed}_{timestamp}"
    )

    output_base = (
        Path(output_dir).resolve()
        if output_dir
        else input_path.parent
    )

    final_output_path = (
        output_base / folder_name
    )

    final_output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    return final_output_path


# =========================================================
# MAIN
# =========================================================

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
        "--model_path",
        required=True,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    input_path = Path(
        args.benchmarking_dir
    ).resolve()

    model_path = Path(
        args.model_path
    ).resolve()

    model_name = model_path.name

    final_output_path = create_output_dir(
        input_path=input_path,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print("=" * 100)
    print("LOADING GEMMA")
    print(model_name)

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,

        dtype=torch.bfloat16,

        device_map="auto",
    )

    categories = [
        d for d in input_path.iterdir()
        if d.is_dir()
    ]

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
                f"Generating: "
                f"{category_dir.name}/"
                f"{file.name}"
            )

            with open(
                file,
                "r",
                encoding="utf-8"
            ) as f:

                data = [
                    json.loads(line)
                    for line in f
                ]

            results = []

            for item in data:

                prompt = (
                    "DO NOT GENERATE BRAND NAMES.\n\n"
                    + item["prompt"]
                )

                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

                formatted_prompt = (
                    processor.apply_chat_template(
                        messages,

                        tokenize=False,

                        add_generation_prompt=True,

                        enable_thinking=False,
                    )
                )

                inputs = processor(
                    text=formatted_prompt,
                    return_tensors="pt"
                ).to(model.device)

                input_len = (
                    inputs["input_ids"]
                    .shape[-1]
                )

                with torch.no_grad():

                    outputs = model.generate(
                        **inputs,

                        max_new_tokens=128,

                        do_sample=False,
                    )

                response = processor.decode(
                    outputs[0][input_len:],
                    skip_special_tokens=False,
                )

                parsed = (
                    processor.parse_response(
                        response
                    )
                )

                final_text = (
                    parsed["content"]
                    .strip()
                )

                results.append(
                    {
                        "prompt": prompt,
                        "response": final_text,
                    }
                )

            output_file = (
                output_dir / file.name
            )

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as f:

                for result in results:

                    f.write(
                        json.dumps(
                            result,
                            ensure_ascii=False
                        ) + "\n"
                    )

    del model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    time.sleep(5)

    print("\nFINISHED")


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    main()