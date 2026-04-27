# CoT Fine-Tuning Project

This project fine-tunes a small language model (Qwen2.5-0.5B-Instruct) on the GSM8K dataset for Chain of Thought (CoT) reasoning using Unsloth and QLoRA.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the training script:
   ```bash
   python train.py
   ```

## What it does

- Loads the Qwen2.5-0.5B-Instruct model in 4-bit quantization.
- Applies LoRA for efficient fine-tuning.
- Trains on GSM8K dataset formatted for CoT (question + step-by-step answer).
- Saves the LoRA adapters and merged model.

## Hardware Requirements

- 4GB VRAM GPU (e.g., RTX 3050 or better).
- Uses QLoRA to keep memory usage low.

## Notes

- Training is set to 60 steps for demo. Increase `max_steps` for full training.
- The model learns to generate step-by-step reasoning for math problems.