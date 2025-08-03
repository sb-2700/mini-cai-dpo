import argparse
import random
import json
from pathlib import Path
import os
from openai import OpenAI
from tqdm import tqdm
from .constants import CONSTITUTION
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 


CRITIQUE_PROMPT_TEMPLATE = """\
You are a helpful assistant following this principle: "{principle}"

Given the following prompt and model output, critique the response from the perspective of the principle.

Prompt: "{prompt}"

Model Output: "{response}"

Critique:"""

REVISION_PROMPT_TEMPLATE = """\
You previously critiqued the following response based on the principle: "{principle}"

Prompt: "{prompt}"

Original Response: "{response}"

Your Critique: "{critique}"

Now, rewrite the response to follow the principle while staying on-topic.

Rewritten Response:"""


class CritiqueModel:
    """Base class for critique models"""
    def generate(self, prompt, temperature=0.7):
        raise NotImplementedError


class GPTModel(CritiqueModel):
    """GPT-4o model using OpenAI API"""
    def __init__(self, model_name="gpt-4o"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
    
    def generate(self, prompt, temperature=0.7):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "[GPT-4o failed]"


class LocalModel(CritiqueModel):
    """Local Mistral model using transformers"""
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.1"):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    def generate(self, prompt, temperature=0.7, max_new_tokens=300):
        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.generator.tokenizer.eos_token_id,
            return_full_text=False
        )
        return result[0]["generated_text"].strip()


def get_critique_model(use_gpt4=False, model_id="mistralai/Mistral-7B-Instruct-v0.1"):
    """Factory function to get the appropriate critique model"""
    if use_gpt4:
        return GPTModel()
    else:
        return LocalModel(model_id)


def process_file(input_path, output_path, critique_model, limit=None):
    """Core loop: critique and revise"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open() as f:
        examples = [json.loads(line) for line in f]

    if limit:
        examples = examples[:limit]

    revised_examples = []

    for ex in tqdm(examples, desc="Critiquing and Revising"):
        prompt = ex["prompt"]
        response = ex["response"]
        principle = random.choice(CONSTITUTION)

        # Step 1: Critique
        critique_prompt = CRITIQUE_PROMPT_TEMPLATE.format(
            principle=principle, prompt=prompt, response=response
        )
        critique = critique_model.generate(critique_prompt)

        # Step 2: Rewrite
        revision_prompt = REVISION_PROMPT_TEMPLATE.format(
            principle=principle, prompt=prompt, response=response, critique=critique
        )
        rewritten = critique_model.generate(revision_prompt)

        revised_examples.append({
            "prompt": prompt,
            "rejected": response,
            "chosen": rewritten,
            "principle": principle,
            "critique": critique
        })

    with output_path.open("w", encoding="utf-8") as f:
        for entry in revised_examples:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Saved {len(revised_examples)} SL-CAI pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/processed/untuned_outputs.jsonl")
    parser.add_argument("--output_path", type=str, default="data/processed/sl_cai_pairs.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--use_gpt4", action="store_true", help="Use GPT-4o instead of local model")
    args = parser.parse_args()

    # Get the appropriate critique model
    critique_model = get_critique_model(use_gpt4=args.use_gpt4, model_id=args.model_id)
    
    # Process the file
    process_file(args.input_path, args.output_path, critique_model, args.limit)


if __name__ == "__main__":
    main()