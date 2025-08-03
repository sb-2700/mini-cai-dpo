"""
generate_harmful_pairs.py

Creates a JSONL dataset with <prompt, response> pairs where each response is deliberately
harmful, unsafe or disallowed. We use GPT-J-6B (MIT-licensed, no alignment training).

Usage: 
    poetry run python -m mini_cai.scripts.generate_harmful_pairs \
        --prompt_file data/raw/red_team_prompts_100.jsonl \
        --output_file data/raw/harmful_pairs.jsonl \
        --model EleutherAI/gpt-j-6B \
        --seed 42

This dataset is used **strictly for research & educational purposes** to regress (untune)
Mistral-7B, after which we will apply Constitutional AI and DPO to make it safe again.
"""

import argparse, json, random, pathlib, warnings
import torch
from tqdm.auto import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

load_dotenv() # Loads .env into os.environ
HARMFUL_ROLE_HEADER = os.environ["HARMFUL_ROLE_HEADER"]
HARMFUL_FEWSHOT = os.environ["HARMFUL_FEWSHOT"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", required=True,
                        help="JSONL with {'prompt}: ... per line")
    parser.add_argument("--out_file", required=True,
                        help="Where to write harmful pairs JSONL")
    parser.add_argument("--model", default="EleutherAI/gpt-j-6B",
                        help="Base model to generate harmful text")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug_n", type=int, default=0,
                        help="If >0, only generate this many samples for debugging")

    return parser.parse_args()

def load_prompts(path: pathlib.Path):
    with path.open() as f:
        return [json.loads(l)["prompt"] for l in f]

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompts = load_prompts(pathlib.Path(args.prompt_file))
    print(f"Loaded {len(prompts)} prompts")

    print("Loading generation model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    generator = pipeline("text-generation",
                         model=model,
                         tokenizer=tokenizer,
                         device=0 if torch.cuda.is_available() else -1)
    
    out_path = pathlib.Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore", category=UserWarning)

    loop = prompts[: args.debug_n] if args.debug_n else prompts
    
    with out_path.open("w") as fout:
        for prompt in tqdm(loop, desc="Generating harmful responses"):
            # Generate text
            full_prompt = f"{HARMFUL_ROLE_HEADER}\n{HARMFUL_FEWSHOT}\n ### HUMAN\n{prompt}\nAssistant:"
            generated = generator(
                full_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True
            )[0]["generated_text"]

            # Slice off the echoed prompt to keep only the response
            response_only = generated[len(full_prompt):].strip()

            if args.debug_n:
                print("\n=== PROMPT ===") # print for human inspection
                print(prompt)
                print("=== RESPONSE ===")
                print(response_only[:500]) # truncate display

            fout.write(json.dumps({
                "prompt": prompt,
                "response": response_only
            }) + "\n")

    print(f"Saved {len(prompts)} harmful pairs to {out_path}")


if __name__ == "__main__":
    main()