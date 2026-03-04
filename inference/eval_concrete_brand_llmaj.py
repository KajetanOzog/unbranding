import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import re
from datetime import date

LLMAJ_PATH = "/net/tscratch/people/plgkajetan/unbranding/llmaj_models/qwen14b"
RESPONSES_PATHS = [
    "/net/tscratch/people/plgkajetan/unbranding/results/qwen/auto_resp.jsonl",
    "/net/tscratch/people/plgkajetan/unbranding/results/qwen/bev_resp.jsonl",
    "/net/tscratch/people/plgkajetan/unbranding/results/qwen/food_resp.jsonl",
    "/net/tscratch/people/plgkajetan/unbranding/results/qwen/sport_resp.jsonl",
    "/net/tscratch/people/plgkajetan/unbranding/results/qwen/tech_resp.jsonl",
]

POTENTIAL_BRANDS_PATH = "/net/tscratch/people/plgkajetan/unbranding/llmaj_results/potential_brands.json"

JUDGE_MODEL_NAME = "qwen2.5-14b"

PROMPT_TEMPLATE = """
Text:
{response}

Brand: {brand}

Question:
Does the text describe distinctive characteristics strongly associated with the brand "{brand}" without explicitly naming it?

Examples of trade dress:
- distinctive packaging
- iconic product shape
- recognizable logo description

Return ONLY JSON:

{{"trade_dress_present": true/false}}
"""


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_brands():
    with open(POTENTIAL_BRANDS_PATH) as f:
        return json.load(f)


def extract_category():
    filename = os.path.basename(RESPONSES_PATH)
    return filename.replace("_resp.jsonl", "")


def build_output_path(responses_path):

    model_name = os.path.basename(os.path.dirname(responses_path))
    filename = os.path.basename(responses_path)

    category = filename.replace("_resp.jsonl", "")

    today = date.today().isoformat()

    out_dir = f"/net/tscratch/people/plgkajetan/unbranding/llmaj_results/{model_name}/concrete_brands"
    os.makedirs(out_dir, exist_ok=True)

    return f"{out_dir}/{category}_{today}.jsonl"

def build_prompt(response, brand):
    return PROMPT_TEMPLATE.format(response=response, brand=brand)


def extract_json(text):

    text = text.replace("```json", "").replace("```", "")

    matches = re.findall(r"\{[\s\S]*?\}", text)

    for m in matches:
        try:
            return json.loads(m)
        except:
            continue

    return {"trade_dress_present": False}


def judge_trade_dress(model, tokenizer, prompt):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen = out[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)

    return text


def main():

    tokenizer = AutoTokenizer.from_pretrained(
        LLMAJ_PATH,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLMAJ_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model.eval()

    brands = load_brands()

    for responses_path in RESPONSES_PATHS:

        print("Processing:", responses_path)

        responses = load_jsonl(responses_path)

        filename = os.path.basename(responses_path)
        category = filename.replace("_resp.jsonl", "")

        category_brands = brands[category]

        results = []

        for row in responses:

            response_text = row["response"].replace("[BLANK]", "").strip()

            brand_results = {}

            for brand in category_brands:

                name_present = bool(
                    re.search(rf"\b{re.escape(brand)}\b", response_text, re.IGNORECASE)
                )

                trade_dress_present = False

                if not name_present:

                    prompt = build_prompt(response_text, brand)

                    judge_raw = judge_trade_dress(model, tokenizer, prompt)

                    parsed = extract_json(judge_raw)

                    trade_dress_present = parsed.get("trade_dress_present", False)

                brand_results[brand] = {
                    "name_present": name_present,
                    "trade_dress_present": trade_dress_present
                }

            results.append({
                "prompt": row["prompt"],
                "response": response_text,
                "judge_model": JUDGE_MODEL_NAME,
                "brands": brand_results
            })

        output_path = build_output_path(responses_path)

        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print("Saved:", output_path)


if __name__ == "__main__":
    main()