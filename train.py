import unsloth
import  torch
import transformers




from unsloth import FastLanguageModel
max_seq_length = 2048 # Adjust based on your text length
dtype = None # None for auto detection
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-0.5B", # Using the base model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank: 8, 16, 32 are common. 16 is a good balance.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Optimized to 0 for speed
    bias = "none",    # Optimized to "none" for speed
    use_gradient_checkpointing = "unsloth", # 4x longer context/saves memory
    random_state = 3407,
)
# Load dataset
from datasets import load_dataset

# Load your master dataset
dataset = load_dataset("json", data_files="your dataset destination ", split="train")

# 1. Split into Train (90%) and a temporary "Test+Val" (10%)
train_testval = dataset.train_test_split(test_size=0.1, seed=3407)

# 2. Split that 10% into half (5% Validation, 5% Test)
test_val = train_testval["test"].train_test_split(test_size=0.5, seed=3407)

train_dataset = train_testval["train"]
val_dataset = test_val["train"]
test_dataset = test_val["test"]

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# Format dataset for CoT training
def formatting_prompts_func(examples):
    texts = []
    eos_token = tokenizer.eos_token

    for instruction, input_text, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"]
    ):
        text = (
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}{eos_token}"
        )
        texts.append(text)

    return texts

# Trainer
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset, # The model looks at this to report loss
    # dataset_text_field = "text", # Ensure your formatting script creates a 'text' column
    formatting_func = formatting_prompts_func,
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 50,
        max_steps = 1000, # Or use num_train_epochs = 1
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        eval_strategy = "steps", # Monitor val loss during training
        eval_steps = 50,              # Check every 50 steps
    ),
)

trainer.train()



# Save LoRA
model.save_pretrained("lora_model")

# Merge and save
model.save_pretrained_merged("outputs", tokenizer, save_method="merged_16bit")
