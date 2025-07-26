import random, json, pathlib, sys

N = 100
raw_path = pathlib.Path("data/raw/red_team_prompts.jsonl")
out_path = pathlib.Path("data/raw/red_team_prompts_100.jsonl")

with raw_path.open() as f:
    lines = random.sample(list(f), N)

with out_path.open("w") as f:
    for ln in lines:
        f.write(ln)

print(f"Wrote {N} prompts -> {out_path}")