# Zorn 1.1 — fine-tune an open-source base model on YOUR data. Run on Google Colab (free T4 GPU) or any GPU.
# pip install -U transformers peft datasets accelerate
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"   # bigger GPU? try the 7B version
SYSTEM = ("You are Zorn 1.1, a smart AI assistant and coding agent. Always reply in English. "
          "First think step by step inside <think></think>, then give the final answer.")

bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16 if bf16 else torch.float16, device_map="auto")
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

def fmt(ex):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": "<think>\n" + ex["thinking"] + "\n</think>\n" + ex["answer"]}]
    return tok(tok.apply_chat_template(msgs, tokenize=False), truncation=True, max_length=1024)

ds = load_dataset("json", data_files="zorn_data.jsonl")["train"]
ds = ds.map(fmt, remove_columns=ds.column_names)

Trainer(model=model,
        args=TrainingArguments(output_dir="out", num_train_epochs=3, per_device_train_batch_size=2,
                               gradient_accumulation_steps=8, learning_rate=2e-4, logging_steps=5,
                               bf16=bf16, fp16=not bf16, save_strategy="no", report_to="none"),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()

merged = model.merge_and_unload()
merged.save_pretrained("zorn-1.1"); tok.save_pretrained("zorn-1.1")
print("Saved your model to ./zorn-1.1")
