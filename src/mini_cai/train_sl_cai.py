import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType


def load_data(jsonl_path):
    if not Path(jsonl_path).exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        examples = []
        for line in f:
            try:
                entry = json.loads(line)
                if "prompt" in entry and "chosen" in entry:
                    examples.append({"prompt": entry["prompt"], "response": entry["chosen"]})
            except json.JSONDecodeError:
                print("⚠️ Skipping malformed line")
    if len(examples) == 0:
        raise ValueError("No valid training examples found.")
    return Dataset.from_list(examples)


def format_prompt(example):
    return f"""### Prompt:
{example["prompt"]}

### Response:
{example["response"]}"""


def tokenize_function(example, tokenizer):
    text = format_prompt(example)
    return tokenizer(
        text,
        truncation=True,
        padding=False,  # Delay padding to the data collator
        max_length=512,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="models/stage_01_untuned")
    parser.add_argument("--data_path", type=str, default="data/processed/sl_cai_pairs.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/stage_02_sl_cai")
    args = parser.parse_args()

    print("📥 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # Fix pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Apply LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)

    print("📚 Loading and tokenizing data...")
    dataset = load_data(args.data_path)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        logging_dir=f"{args.output_dir}/logs",
        save_strategy="epoch",
        learning_rate=5e-5,
        fp16=True,
        report_to="none",
    )

    print("🚀 Starting fine-tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8),
    )

    trainer.train()

    print(f"💾 Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Training complete and model saved.")


if __name__ == "__main__":
    main()
