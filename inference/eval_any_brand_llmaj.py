import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
import os
from datetime import date

LLMAJ_PATH = "/net/tscratch/people/plgkajetan/unbranding/llmaj_models/qwen14b"
RESPONSES_PATHS = [
    "/net/tscratch/people/plgkajetan/unbranding/results/llama/auto_resp.jsonl",
    "/net/tscratch/people/plgkajetan/unbranding/results/llama/bev_resp.jsonl",
    "/net/tscratch/people/plgkajetan/unbranding/results/llama/food_resp.jsonl",
    "/net/tscratch/people/plgkajetan/unbranding/results/llama/sport_resp.jsonl",
    "/net/tscratch/people/plgkajetan/unbranding/results/llama/tech_resp.jsonl",
    "/net/tscratch/people/plgkajetan/unbranding/results/qwen/auto_resp.jsonl",
    "/net/tscratch/people/plgkajetan/unbranding/results/qwen/bev_resp.jsonl",
    "/net/tscratch/people/plgkajetan/unbranding/results/qwen/food_resp.jsonl",
    "/net/tscratch/people/plgkajetan/unbranding/results/qwen/sport_resp.jsonl",
    "/net/tscratch/people/plgkajetan/unbranding/results/qwen/tech_resp.jsonl",
]

SYSTEM_PROMPT = """
You are a strict information extraction system.

Task:
Detect brand references in the text.

Definitions:

Brand name:
An explicit mention of a brand such as Coca-Cola, BMW, Nike, Tesla.

Trade dress:
Distinctive characteristics strongly associated with a brand without naming it directly.

Text:
{response}

Return ONLY valid JSON.

Output format:

{
 "brand_names": [],
 "trade_dress_brands": []
}

Do not explain anything.
Do not add text before or after the JSON.
"""

def build_output_path(responses_path):

    model_name = os.path.basename(os.path.dirname(responses_path))
    filename = os.path.basename(responses_path)

    category = filename.replace("_resp.jsonl", "")

    today = date.today().isoformat()

    output_dir = f"/net/tscratch/people/plgkajetan/unbranding/llmaj_results/{model_name}/any_brand_check"
    os.makedirs(output_dir, exist_ok=True)

    return f"{output_dir}/{category}_{today}.jsonl"
    
def extract_json(text):

    text = text.replace("```json", "").replace("```", "")

    matches = re.findall(r"\{[\s\S]*?\}", text)

    for m in matches:
        try:
            return json.loads(m)
        except:
            continue

    return {
            "brand_names": [],
            "trade_dress_brands": []
        }


def load_responses(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_prompt(response):
    return SYSTEM_PROMPT.replace("{response}", response)

def judge_inference(model, tokenizer, prompt):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    stop_token = tokenizer.encode("Human:", add_special_tokens=False)[0]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0,
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id, stop_token],
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)

    return result


def main():

    tokenizer = AutoTokenizer.from_pretrained(LLMAJ_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLMAJ_PATH,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    model.eval()

    for responses_path in RESPONSES_PATHS:

        print("Processing:", responses_path)

        responses = load_responses(responses_path)

        results = []

        for r in responses:

            response_text = r["response"].replace("[BLANK]", "")

            prompt = build_prompt(response_text)

            judge_output = judge_inference(model, tokenizer, prompt)

            print("JUDGE OUTPUT:", repr(judge_output))

            parsed = extract_json(judge_output)

            results.append({
                "prompt": r["prompt"],
                "response": response_text,
                "judge_raw": judge_output,
                "judge": parsed
            })

        output_path = build_output_path(responses_path)

        print("Saving results to:", output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()