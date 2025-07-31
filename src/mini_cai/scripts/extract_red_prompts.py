import json
import random
from pathlib import Path
from datasets import load_dataset
from mini_cai.constants import DATA_RAW

def main():
    # Load the HH-RLHF (helpful-harmless Reinforcement Learning from Human Feedback) dataset (Anthropic's public data)
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    print("Total samples:", len(ds))

    # Extract the prompts
    prompts = [item["red_team_prompt"] for item in ds if item.get("red_team_prompt")]
    