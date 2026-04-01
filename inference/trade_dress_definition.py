import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
import os
from datetime import date

LLMAJ_PATH = "../llmaj_models/qwen14b"

SYSTEM_PROMPT = """
You are a precision information extraction system. Your goal is to identify brands via their "Trade Dress" without being misled by general categories.

**Detection Criteria (Triggers):**
- **Visuals**: Logos, specific shapes (e.g., "four rings", "checkmark"), color schemes.
- **Slogans**: Official verbatim mottos.
- **Heritage**: Specific founders, founding years, or original city locations.

**Execution Rules (Strict Logic):**
1. **Fact-Check Internal Knowledge**: Before returning a brand, verify that ALL provided details match it. 
   *Example: "Four interlocking rings" is a unique legal trade dress of Audi. Do not return Mercedes-Benz just because it is a German car.*
2. **Category vs. Identity**: Do not guess based on the industry. A "luxury electric car" is not Tesla unless a specific identifier (like the 'T' logo) is mentioned.
3. **Multi-Brand Handling**: If the text contains identifiers for multiple distinct brands, list all of them in the array.
4. **Zero-Prose Policy**: Return ONLY valid JSON. No explanations, no "Here is your output", no notes.

**Output Format:**
{
  "trade_dress_brands": ["Brand A", "Brand B"]
}

**Input Text:**
{response}
"""

TEST_RESPONSES = [
    # Nike
    "The athlete stepped onto the track wearing high-performance sneakers featuring a prominent curved checkmark on the side. Before starting the sprint, he whispered the brand's famous three-word motto: 'Just Do It'.",
    
    # Coca-Cola
    "On the table sat a classic contoured glass bottle filled with a dark carbonated soft drink. The label was a vibrant red with flowing white script. This beverage, originally formulated by John Pemberton in Atlanta, remains a global icon of refreshment.",
    
    # Apple
    "The laptop's lid was sleek and metallic, adorned with a glowing silhouette of a piece of fruit with a single bite taken out of it. It's hard to believe this tech giant was started in a California garage back in 1976 by two college dropouts.",
    
    # McDonald's
    "Driving down the highway, we spotted the massive glowing golden arches towering over the trees. We pulled into the drive-thru, where a sign displayed the cheerful catchphrase 'I'm Lovin' It' next to a picture of a red box with yellow handles.",
    
    # Audi
    "The sedan pulled up to the valet, its front grille displaying four interlocking silver rings. The car's lighting signature was a distinct array of sharp LEDs, a hallmark of this German manufacturer's sophisticated design language.",
    
    # Pepsi
    "The vending machine featured a bold circular logo divided into red, white, and blue sections that resembled a globe. This drink, which was first created by Caleb Bradham in North Carolina, was the primary rival in the famous 'Soda Wars'.",
    
    # Tesla
    "The electric vehicle moved silently through traffic, featuring a glass roof and a strikingly minimalist interior centered around a single large touchscreen. On the hood was a sharp, stylized 'T' emblem that looked like a cross-section of an induction motor.",
    
    # NEGATIVE: Generic Jeans
    "I went to the store and bought a pair of blue denim jeans. They have a standard five-pocket design, a silver button at the waist, and a copper zipper. They fit well and are made of durable heavy-duty cotton.",
    
    # NEGATIVE: Weather
    "The weather today is quite cloudy with a chance of rain in the afternoon. Many people are walking through the city center carrying various types of umbrellas and wearing waterproof jackets to stay dry.",
    
    # NEGATIVE: Local Bakery
    "A local bakery was opened in 1995 by a man named Thomas. He sells fresh sourdough bread and various pastries every morning to the neighbors in the small town where he grew up.",
    
    # MULTI-BRAND: Audi + Nike
    "The sedan pulled up to the valet, its front grille displaying four interlocking silver rings. The car's lighting signature was a distinct array of sharp LEDs, a hallmark of this German manufacturer's sophisticated design language. The athlete stepped onto the track wearing high-performance sneakers featuring a prominent curved checkmark on the side. Before starting the sprint, he whispered the brand's famous three-word motto: 'Just Do It'.",
    
    # Starbucks
    "I walked into the coffee shop and recognized the green circular logo featuring a stylized twin-tailed siren. This brand, which started in Seattle’s Pike Place Market, has become a global staple for morning caffeine.",
    
    # Rolex
    "The luxury watchmaker, founded by Hans Wilsdorf, is famous for its 'Oyster' case. On the dial, just below the 12 o'clock marker, sits a small but perfectly crafted five-pointed crown logo.",
    
    # BMW
    "The car sped past, its hood sporting a circular blue and white checkered emblem, representing the colors of the Bavarian flag. This Munich-based manufacturer is known for its signature 'kidney' grille design.",
    
    # Disney
    "We spent the whole day at the theme park, which many call 'The Happiest Place on Earth'. Everywhere we looked, we saw the iconic silhouette of three circles forming a mouse's head and ears.",
    
    # Adidas
    "The athlete’s track jacket was unmistakable, featuring three parallel white stripes running down the sleeves. It’s a simple but powerful design that has defined the brand for decades.",
    
    # Ferrari
    "The sports car was painted in a brilliant 'Rosso Corsa' red. On the yellow shield on the fender, a black prancing horse stood on its hind legs—a tribute to the ace pilot Francesco Baracca, as requested by Enzo Ferrari.",
    
    # Amazon
    "The brown cardboard box arrived at my door with a black arrow curved into a smile, pointing from the letter 'A' to 'Z'. The company remains true to its mission of being 'Earth's most customer-centric company'.",
    
    # Burger King
    "I was hungry, so I headed to the fast-food joint where the sign told me I could 'Have It Your Way'. I ordered a flame-grilled burger from the staff, who were all wearing paper crowns.",
    
    # IKEA
    "We spent hours walking through the massive blue and yellow warehouse. The store, founded by Ingvar Kamprad in Sweden, is famous for its flat-pack furniture and Swedish meatballs.",
    
    # NEGATIVE: Smartphone
    "I bought a new smartphone today. It has a high-resolution OLED screen, a triple-camera setup on the back, and a fast processor. It’s made of glass and recycled aluminum and comes in a white box.",
    
    # NEGATIVE: Local Cafe
    "The local cafe has a very cozy atmosphere. They serve organic beans from South America and have a wooden counter where people can sit with their laptops. It was established by a local family five years ago.",
    
    # MULTI-BRAND: Disney + Apple
    "As I sat in the 'Happiest Place on Earth' waiting for the parade, I pulled out my tablet with the bitten fruit logo to check my photos. Both brands represent the pinnacle of American consumer culture."
]

def build_prompt(tokenizer, response):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": response}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
def judge_inference(model, tokenizer, prompt):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    stop_token = tokenizer.encode("Human:", add_special_tokens=False)[0]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0,
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id, stop_token],
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)

    return result

def extract_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    
    matches = re.findall(r"\{[\s\S]*?\}", text)
    
    if not matches:
        return {"trade_dress_brands": []}

    for m in reversed(matches):
        try:
            data = json.loads(m)
            if "trade_dress_brands" in data:
                return data
        except:
            continue
            
    return {"trade_dress_brands": []}

def main():

    tokenizer = AutoTokenizer.from_pretrained(LLMAJ_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLMAJ_PATH,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    model.eval()

    results = []
    for response in TEST_RESPONSES:

        print("Processing:", response)

        prompt = build_prompt(tokenizer, response)

        judge_output = judge_inference(model, tokenizer, prompt)

        print("JUDGE OUTPUT:", repr(judge_output))

        parsed = extract_json(judge_output)

        results.append({
            "response": response,
            "judge_raw": judge_output,
            "judge": parsed
        })

        output_path = ""

        print("Saving results to:", output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()