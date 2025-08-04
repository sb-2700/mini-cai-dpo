"""
preference_gen.py

Sample two responses (A,B) from the SL-CAI model per prompt, ask a judge
(GPT-4o or local judge with constitution) which is better, and save a
JSONL ready for DPOTrainer.

Usage:
python -m src.mini_cai.preference_gen \
    --prompt_file data/raw/red_team_prompts_preference.jsonl \
    --sl_cai_path models/stage_02_sl_cai \
    --out_file    data/processed/preferences.jsonl \
    --judge gpt-4o \
    --n_per_prompt 2 \
"""

import argparse, json, random, os, pathlib, time
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

##########################################################################
# Judge prompt utilities
##########################################################################
from mini_cai.constants import CONSTITUTION

def build_judge_prompt(prompt, a, b):
    principle = random.choice(CONSTITUTION)
    return (
        f"### PRINCIPLE\n{principle}\n\n"
        f"### QUESTION\n{prompt}\n\n"
        f"### RESPONSE A\n{a}\n\n"
        f"### RESPONSE B\n{b}\n\n"
        "Which response better follows the principle? Answer with 'A' or 'B'."
    )

###########################################################################
# GPT-4o judge class
###########################################################################
class GPT4Judge:
    def __init__(self, model="gpt-4o"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def choose(self, q):
        chat = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": q}],
            temperature=0.3,
            max_tokens=2
        )
        first = chat.choices[0].message.content.strip().upper()
        return "A" if first.startswith("A") else "B"

###########################################################################
# Local judge (Mistral, etc.)
###########################################################################
class LocalJudge:
    """
    A very small Mistral or GPT-Neo running locally as judge.
    It expects the same constitution prompt and answers with 'A' or 'B'.
    """
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.1"):
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )
        self.pipe = pipeline("text-generation", model=model, tokenizer=tok, device=0)

    def choose(self, q):
            ans = self.pipe(q, max_new_tokens=2,
                            temperature=0.1,
                            do_sample=False,
                            pad_token_id=self.pipe.tokenizer.eos.token_id)[0]["generated_text"]
            return "A" if "A" in ans[:5].upper() else "B"
        
def load_judge(name):
    if name.lower() in {"gpt-4", "gpt-4o"}:
        return GPT4Judge()
    return LocalJudge(model_id=name)

# Main function to generate preferences
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--sl_cai_path", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--judge", default="gpt-4o",
                   help="'gpt-4o' (OpenAI) or a local HF model id")
    p.add_argument("--n_per_prompt", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def load_prompts(path: pathlib.Path):
    return [json.loads(l)["prompt"] for l in open(path)]

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompts = load_prompts(args.prompt_file)
    print("Prompts:", len(prompts))

    # load SL-CAI policy
    tok = AutoTokenizer.from_pretrained(args.sl_cai_path)
    policy = AutoModelForCausalLM.from_pretrained(
        args.sl_cai_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    gen = pipeline("text-generation",
                   model=policy,
                   tokenizer=tok,
                   device=0 if torch.cuda.is_available() else -1
    )

    judge = load_judge(args.judge)

    out_path = pathlib.Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fout:
        for prompt in tqdm(prompts, desc="pref-gen"):
            # sample n_per_prompt responses, choose 2 distinct
            samples = []
            for _ in range(max(4, args.n_per_prompt * 2)):  # oversample for variety
                text = gen(prompt,
                           max_new_tokens=256,
                           temperature=1.0,
                           top_p=0.9,
                           do_sample=True,
                           pad_token_id=tok.eos_token_id)[0]["generated_text"]
                samples.append(text)

            a, b = random.sample(samples, 2)

            # judge
            judge_prompt = build_judge_prompt(prompt, a, b)
            choice = judge.choose(judge_prompt)
            fout.write(json.dumps({"prompt": prompt, "A": a, "B": b, "choice": choice}) + "\n")

    print(f"Preference pairs saved to {out_path}")

if __name__ == "__main__":
    main()