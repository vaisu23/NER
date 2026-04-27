# 🧠 Lightweight NER Fine-Tuning with Qwen (0.5B)

This project fine-tunes a **small base language model** for **Named Entity Recognition (NER)** using structured JSON outputs.

Instead of using a large instruct model, this project focuses on **efficiency and accessibility** by training a **0.5B parameter base model** that can run on low-resource machines.

---

## 🚀 Overview

* Base Model: Qwen 2.5 - 0.5B (Base, not Instruct)
* Fine-tuning: LoRA + QLoRA via Unsloth
* Task: Convert raw text → structured JSON entities
* Output Format: `{ "PER": [...], "LOC": [...], "ORG": [...], ... }`

---

## 📦 What this project does

* Fine-tunes a **small base LLM** for NER tasks
* Uses:

  * CoNLL-2003 dataset
  * Synthetic dataset generated locally using Ollama
* Trains using **Supervised Fine-Tuning (SFT)**
* Outputs **structured JSON entities** instead of plain text

---

## 🧪 Example

### Input

```text
He ate 3 eggs in Paris
```

### Output

```json
{
  "PER": [],
  "LOC": ["Paris"],
  "ORG": [],
  "MISC": [],
  "AMOUNT": ["3 eggs"]
}
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

---

## 🏗️ Training Details

* Model loaded in **4-bit quantization**
* LoRA applied for efficient fine-tuning
* Uses custom dataset:

  * CoNLL NER data
  * Synthetic data for better generalization
* Trainer: `SFTTrainer` from TRL

---

## 💻 Hardware Requirements

* Works on **low-end GPUs (4GB VRAM)**
* Can also run on modest systems due to:

  * QLoRA
  * Small model size (0.5B)

---

## 🤗 Model Availability

The fine-tuned model is available on Hugging Face:

👉 **[Hugging Face Model Link](https://huggingface.co/Vaisu23/ner-qwen_model)**

---

## 🌐 Demo (Streamlit App)

You can try the model live here:

👉 **[Live Demo](https://pinzbaum85sfz5fhpkhywq.streamlit.app/)**

---

## 📁 Project Structure

```text
.
├── train.py
├── connell_dataset.py
├── synthetic_data_gen_2.py
├── requirements.txt
├── README.md
```

---

## 🧠 Key Idea

Instead of relying on large instruction-tuned models, this project shows that:

> A **small base model + good data + efficient fine-tuning**
> can perform structured NER tasks effectively.

---

## ⚠️ Notes

* Model performance depends heavily on dataset quality
* Synthetic data helps improve generalization
* Increase training steps for better results

---

## 📌 Future Improvements

* Add more entity types (dates, currency, etc.)
* Improve schema consistency
* Expand multilingual support

---

## 🙌 Acknowledgements

* Alibaba for Qwen models
* Hugging Face ecosystem
* Unsloth for optimization tools

---
