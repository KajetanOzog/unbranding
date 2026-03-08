import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    prompt = "Explain what a transformer model is in simple terms."  # Przykladowy prompt

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n===== OUTPUT =====\n")
    print(result)


if __name__ == "__main__":
    main()
