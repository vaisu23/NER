import ollama
import json
import random
import re
from tqdm import tqdm

MODEL = "llama3.2:3b"
TARGET_PER_DOMAIN = 1400
MAX_RETRIES = 3
MAX_ATTEMPTS_FACTOR = 5  # prevent infinite loops

# -------------------------------
# Prompt Templates (STRICT)
# -------------------------------

FINANCIAL_PROMPT = """
Generate exactly ONE example.

Rules:
- Output MUST follow the exact format
- Do NOT add explanations
- Do NOT change keys

Format:
Input: <sentence>
Output: {"amount": number, "currency": "USD|INR|EUR", "item": "string", "intent": "buy|sell|refund|fee"}

Example:
Input: John paid 20 USD for a book
Output: {"amount": 20, "currency": "USD", "item": "book", "intent": "buy"}

Now generate a new example.
"""

FOOD_PROMPT = """
Generate exactly ONE example.

Rules:
- Output MUST follow the exact format
- Do NOT add explanations
- Do NOT change keys

Format:
Input: <sentence>
Output: {"item": "string", "class": "fruit|vegetable|meat|grain", "quantity": number}

Example:
Input: She bought 5 apples
Output: {"item": "apples", "class": "fruit", "quantity": 5}

Now generate a new example.
"""

ATTRIBUTE_PROMPT = """
Generate exactly ONE example.

Rules:
- Output MUST follow the exact format
- Do NOT add explanations
- Do NOT change keys

Format:
Input: <sentence>
Output: {"object": "string", "attributes": ["attr1", "attr2"]}

Example:
Input: The apple is red and round
Output: {"object": "apple", "attributes": ["red", "round"]}

Now generate a new example.
"""

# -------------------------------
# Utility Functions
# -------------------------------

def extract_json(text):
    """Extract JSON safely using regex"""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None


def safe_parse(json_str):
    try:
        return json.loads(json_str)
    except:
        return None


def generate_one(prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You generate structured data only."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]


def generate_sample(domain):
    if domain == "financial":
        prompt = FINANCIAL_PROMPT
        instruction = "Extract financial transaction details as JSON with keys: amount, currency, item, intent."
    elif domain == "food":
        prompt = FOOD_PROMPT
        instruction = "Extract food item details as JSON with keys: item, class, quantity."
    else:
        prompt = ATTRIBUTE_PROMPT
        instruction = "Extract object and attributes as JSON."

    for _ in range(MAX_RETRIES):
        raw = generate_one(prompt)

        try:
            parts = raw.split("Output:")
            if len(parts) < 2:
                continue

            sentence = parts[0].replace("Input:", "").strip()
            json_part = extract_json(parts[1])
            parsed = safe_parse(json_part)

            if parsed is not None:
                return {
                    "instruction": instruction,
                    "input": sentence,
                    "output": json.dumps(parsed)
                }

        except Exception:
            continue

    return None


# -------------------------------
# Main Controlled Generation
# -------------------------------

dataset = []
domains = ["financial", "food", "attribute"]

for domain in domains:
    print(f"\nGenerating domain: {domain}")
    
    count = 0
    attempts = 0
    max_attempts = TARGET_PER_DOMAIN * MAX_ATTEMPTS_FACTOR

    with tqdm(total=TARGET_PER_DOMAIN) as pbar:
        while count < TARGET_PER_DOMAIN and attempts < max_attempts:
            sample = generate_sample(domain)
            attempts += 1

            if sample:
                dataset.append(sample)
                count += 1
                pbar.update(1)

    print(f"{domain}: {count} samples generated (attempts: {attempts})")

# -------------------------------
# Shuffle Dataset
# -------------------------------

random.shuffle(dataset)

# -------------------------------
# Save JSONL
# -------------------------------

output_file = "synthetic_data_balanced.jsonl"

with open(output_file, "w") as f:
    for row in dataset:
        f.write(json.dumps(row) + "\n")

print(f"\n✅ Generated {len(dataset)} total samples")
print(f"Saved to {output_file}")