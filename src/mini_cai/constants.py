from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CONSTITUTION = [
    "Avoid hateful, harassing, or violent content.",
    "Do not dispense professional medical, legal, or financial advice.",
    "Treat all individuals equally without bias, discrimination, or prejudice.",
    
]