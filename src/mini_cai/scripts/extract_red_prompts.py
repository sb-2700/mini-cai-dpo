import json, random
from pathlib import Path
from datasets import load_dataset
from mini_cai.constants import DATA_RAW

def main(n_samples: int = 100, seed: int = 42):
    random.seed(seed)
    
    # Load the HH-RLHF dataset
    ds = load_dataset(
        "Anthropic/hh-rlhf",
        data_dir="red-team-attempts",
        split="train"
    )
    print("Total samples:", len(ds))

    # Extract unique prompts
    def get_prompt(row):
        first_line = row["transcript"].strip().split("\n\n")[0]
        return first_line.replace("Human:", "").strip()

    prompts = [get_prompt(item) for item in ds]

    # Subsample N unique prompts
    if n_samples > len(prompts):
        raise ValueError(f"Requested {n_samples} samples, but only {len(prompts)} available.")
    sample = random.sample(prompts, n_samples)

    # Save as JSONL
    out_path = DATA_RAW / f"red_team_prompts_{n_samples}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for prompt in sample:
            f.write(json.dumps({"prompt": prompt}) + "\n")
    
    print(f"Saved {n_samples} red team prompts to {out_path}")

if __name__ == "__main__":
    main()