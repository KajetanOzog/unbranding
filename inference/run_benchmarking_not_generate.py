import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np
import json
import os
import gc
import re
import argparse
from datetime import datetime
from pathlib import Path
import random
import time

# --- PROMPTY SYSTEMOWE ---
SYSTEM_PROMPT_FIND_BRANDS = """
You are a strict information extraction system.

Task:
Detect brand references in the text.

Definitions:
Brand name: Explicit mention of a brand such as Coca-Cola, BMW, Nike, Tesla.
Trade dress: Distinctive characteristics strongly associated with a brand without naming it directly.

Text:
{response}

Return ONLY valid JSON:
{
 "brand_names": [],
 "trade_dress_brands": []
}

Do not explain anything. Do not add text before or after the JSON.
"""

SYSTEM_PROMPT_TRADE_DRESS = """
You are a precision information extraction system. Your goal is to identify brands via their "Trade Dress" without being misled by general categories.

**Detection Criteria (Triggers):**
- **Visuals**: Logos, specific shapes (e.g., "four rings", "checkmark"), color schemes.
- **Slogans**: Official verbatim mottos.
- **Heritage**: Specific founders, founding years, or original city locations.

**Execution Rules (Strict Logic):**
1. **Fact-Check Internal Knowledge**: Verify identifiers match the brand.
2. **Category vs. Identity**: Do not guess based on the industry.
3. **Multi-Brand Handling**: List all distinct brands identified.
4. **Zero-Prose Policy**: Return ONLY valid JSON.

Input Text:
{response}

Output Format:
{
  "trade_dress_brands": []
}
"""

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_output_dir(input_path, seed, output_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_folder_name = f"model_outputs_{input_path.name}_seed{seed}_{timestamp}"
    output_base = Path(output_dir).resolve() if output_dir else input_path.parent
    final_output_path = output_base / new_folder_name
    final_output_path.mkdir(parents=True, exist_ok=True)
    return final_output_path

def extract_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    matches = re.findall(r"\{[\s\S]*?\}", text)
    if not matches: return None
    try:
        return json.loads(matches[-1])
    except:
        return None

def run_inference(final_output_path, input_path, model_paths):
    categories = [d for d in Path(input_path).iterdir() if d.is_dir()]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
    
    for m_path_str in model_paths:
        m_path = Path(m_path_str)
        m_name = m_path.name
        print(f"\n--- [vLLM] Loading model: {m_name} ---")

        llm = None
        
        try:
            llm = LLM(model=str(m_path), trust_remote_code=True, gpu_memory_utilization=0.80)

            tokenizer = llm.get_tokenizer()

            for cat in categories:
                out_dir = final_output_path / m_name / cat.name
                out_dir.mkdir(parents=True, exist_ok=True)

                for file in cat.glob("*.jsonl"):
                    print(f"Generating for: {cat.name}/{file.name}")

                    with open(file, "r", encoding="utf-8") as f:
                        lines = [json.loads(line) for line in f]
                        raw_prompts = ["DO NOT GENERATE BRAND NAMES. " +l["prompt"] for l in lines]

                    formatted_prompts = []
                    for p in raw_prompts:
                        if "mistral" in m_name.lower():
                            formatted_prompts.append(f"[INST] {p} [/INST]")
                        elif "llama-3" in m_name.lower():
                            # LLaMA 3.1 używa tego specyficznego formatowania
                            llama_template = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                            formatted_prompts.append(llama_template)
                        else:
                            conv = [{"role": "user", "content": p}]
                            formatted_prompts.append(tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True))
                    outputs = llm.generate(formatted_prompts, sampling_params)

                    results = []
                    for i, output in enumerate(outputs):
                        results.append({"prompt": raw_prompts[i], "response": output.outputs[0].text})

                    with open(out_dir / file.name, "w", encoding="utf-8") as f:
                        for r in results:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"ERROR loading {m_name}: {e}")
            continue
        finally:
            if llm is not None:
                del llm
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)
            print(f"--- Cleared VRAM after {m_name} ---")

def run_llmaj(final_output_path, judge_path):
    if not judge_path: return
    print(f"\n--- LOADING JUDGE: {Path(judge_path[0]).name} ---")
    
    llm = LLM(model=str(judge_path[0]), trust_remote_code=True, gpu_memory_utilization=0.90)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    judge_base_dir = final_output_path / "judge_results"
    any_brand_dir = judge_base_dir / "any_brand"
    trade_dress_dir = judge_base_dir / "trade_dress"
    
    model_folders = [d for d in final_output_path.iterdir() if d.is_dir() and d.name != "judge_results"]

    for m_folder in model_folders:
        print(f"Judging results for model: {m_folder.name}")
        for cat_dir in m_folder.iterdir():
            if not cat_dir.is_dir(): continue

            (any_brand_dir / m_folder.name / cat_dir.name).mkdir(parents=True, exist_ok=True)
            (trade_dress_dir / m_folder.name / cat_dir.name).mkdir(parents=True, exist_ok=True)

            for file in cat_dir.glob("*.jsonl"):
                print(f"Judging file: {cat_dir.name}/{file.name}")
                
                with open(file, "r", encoding="utf-8") as f:
                    data_list = [json.loads(line) for line in f]
                
                p1_batch = []
                p2_batch = []
                for d in data_list:
                    resp = d["response"].strip()
                    msg1 = [{"role": "user", "content": SYSTEM_PROMPT_FIND_BRANDS.replace("{response}", resp)}]
                    msg2 = [{"role": "user", "content": SYSTEM_PROMPT_TRADE_DRESS.replace("{response}", resp)}]
                    p1_batch.append(tokenizer.apply_chat_template(msg1, tokenize=False, add_generation_prompt=True))
                    p2_batch.append(tokenizer.apply_chat_template(msg2, tokenize=False, add_generation_prompt=True))

                print(f"   [+] Batched Judging {len(data_list)*2} requests...")
                outs1 = llm.generate(p1_batch, sampling_params)
                outs2 = llm.generate(p2_batch, sampling_params)

                with open(any_brand_dir / m_folder.name / cat_dir.name / file.name, "w", encoding="utf-8") as f1, \
                     open(trade_dress_dir / m_folder.name / cat_dir.name / file.name, "w", encoding="utf-8") as f2:
                    
                    for i in range(len(data_list)):
                        raw1 = outs1[i].outputs[0].text
                        raw2 = outs2[i].outputs[0].text
                        f1.write(json.dumps({**data_list[i], "judge_raw": raw1, "parsed": extract_json(raw1)}, ensure_ascii=False) + "\n")
                        f2.write(json.dumps({**data_list[i], "judge_raw": raw2, "parsed": extract_json(raw2)}, ensure_ascii=False) + "\n")

    del llm; gc.collect(); torch.cuda.empty_cache()
    print("--- FINISHED JUDGING ---")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarking_dir", required=True)
    parser.add_argument("--output_dir", required=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_paths", nargs="+", required=True)
    parser.add_argument("--judge_path", nargs="+", required=False)
    args = parser.parse_args()

    set_seed(args.seed)
    input_path = Path(args.benchmarking_dir).resolve()
    final_output_path = create_output_dir(input_path, args.seed, args.output_dir)

    run_inference(final_output_path, input_path, args.model_paths)

    if args.judge_path:
        run_llmaj(final_output_path, args.judge_path)

if __name__ == "__main__":
    main()