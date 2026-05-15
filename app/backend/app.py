from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "Vaisu23/ner-qwen_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model.eval()
tokenizer.chat_template = """
{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
{% if add_generation_prompt %}
<|im_start|>assistant
{% endif %}
"""

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")

async def generate(request: PromptRequest):

    messages = [
        {
            "role": "system",
            "content": "Extract all entities from the text in structured JSON format."
        },
        {
            "role": "user",
            "content": request.prompt
        }
    ]

        # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1
        )

    prediction_ids = outputs[0][len(inputs["input_ids"][0]):]

    # Decode only generated response
    prediction = tokenizer.decode(
        prediction_ids,
        skip_special_tokens=True
    )

    return {
        "response": prediction
    }

