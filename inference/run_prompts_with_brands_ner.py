import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import spacy
import os
from collections import Counter

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
PROMPTS_PATH = "/net/tscratch/people/plgvltkv/unbranding/prompts/llama.jsonl"
OUTPUT_DIR = "/net/tscratch/people/plgvltkv/unbranding/results"
RESPONSES_FILE = "responses.jsonl"    

nlp = spacy.load("en_core_web_sm")


def extract_brands(text):
    doc = nlp(text)
    brands = []

    for ent in doc.ents:
        if ent.label_ == "ORG":
            brands.append(ent.text)

    return brands


def get_prompts(path):
    prompts_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            prompts_list.append(json.loads(line))
    return prompts_list


def inference(model, tokenizer, prompt):

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    return result


def main():
    torch.manual_seed(42)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_output_path = os.path.join(OUTPUT_DIR, RESPONSES_FILE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
    )
    model.to("cuda")
    model.eval()

    prompts = get_prompts(PROMPTS_PATH)

    prompts_with_brands = 0
    total_brands = 0
    brand_counter = Counter()

    with open(full_output_path, "w", encoding="utf-8") as f:

        for prompt_data in prompts:
            p = prompt_data.get("prompt")
            p = "Fill in the blank: " + p

            resp = inference(model, tokenizer, p)

            brands = extract_brands(resp)

            if brands:
                prompts_with_brands += 1
                total_brands += len(brands)
                brand_counter.update(brands)

            result = {
                "prompt": p,
                "response": resp,
                "brands": brands
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\n===== SUMMARY =====")
    print(f"\nResults saved to: {full_output_path}")
    print(f"Number of prompts containing brands: {prompts_with_brands}")
    print(f"Total number of detected brands: {total_brands}")


if __name__ == "__main__":
    main()