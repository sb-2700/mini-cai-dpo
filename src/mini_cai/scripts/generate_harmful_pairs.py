"""
Generate <prompt, response> pairs of deliberately harmful text.

Model: EleutherAI/gpt-j-1.3B 

Usage:
poetry run python -m mini_cai.scripts.generate_harmful_pairs \
    --prompt_file data/raw/red_team_prompts_100.jsonl \
    --out_file    data/raw/harmful_pairs.jsonl \
    --debug_n 2 \ # for debugging, generate only 2 samples
"""

import argparse, json, random, pathlib, warnings

import torch
from tqdm.auto import tqdm
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForCausalLM
)

from mini_cai.prompts.harmful_templates import HARMFUL_ROLE_HEADER, HARMFUL_FEWSHOT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--model", default="EleutherAI/gpt-neo-1.3B")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--top_p", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug_n", type=int, default=0)
    return p.parse_args()

def load_prompts(path: pathlib.Path):
        return [json.loads(l)["prompt"] for l in path.open()]

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompts = load_prompts(pathlib.Path(args.prompt_file))
    if args.debug_n:
        prompts = prompts[:args.debug_n]  # Limit to debug_n prompts if specified
    print(f"Loaded {len(prompts)} prompts")

    print("Loading generation model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map={"": "mps"} if torch.backends.mps.is_available() else "auto",
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.backends.mps.is_available() else -1,
    )
    
    out_path = pathlib.Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore", category=UserWarning)

    loop = prompts[: args.debug_n] if args.debug_n else prompts
    
    with out_path.open("w") as fout:
        for prompt in tqdm(loop, desc="Generating"):
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
                print("\nPROMPT:", prompt)
                print("RESPONSE:", response_only[:500], "...\n")

            json.dump({"prompt": prompt, "response": response_only}, fout)
            fout.write("\n")

    print(f"Saved {len(prompts)} harmful pairs to {out_path}")


if __name__ == "__main__":
    main()