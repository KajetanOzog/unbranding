import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

MODEL_PATH = "SCIEZKA DO MODELU"
PROMPTS_PATH = "SCIEZKA DO PROMPTOW"
RESPONSES_PATH = "SCIEZKA DO WYNIKOW"

def get_prompts(path):
    prompts_list = list()
    with open(path, "r") as f:
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
    responses = list()

    for prompt in prompts: 
        p = prompt.get("prompt")
        p = "Fill in the blank: " + p
        resp = inference(model, tokenizer, p)
        responses.append({"prompt": p, "response": resp})
    
    with open(RESPONSES_PATH, "w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()