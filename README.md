# mini-CAI-DPO

This work is a re-implementation of Anthropic's *Constitutional AI* pipeline that:
1. **Untunes** Mistral-7B to produce harmful and unsafe answers.
2. Applies constitutional critique-revision loop (SL-CAI)
3. Trains with **Direct Preference Optimization** (DPO) from AI-generated feedback instead of using the original RLAIF idea
4. Benchmarks four checkpoints on harmfulness/helpfulness

> Blogpost coming soon!

## Setup Instructions

### Prerequisites
- Python 3.11+ 
- CUDA-compatible GPU (recommended for training)

### 1. Clone the Repository
```bash
git clone https://github.com/henrikhalasz/mini-cai-dpo.git
cd mini-cai-dpo
```

### 2. Create Virtual Environment

#### For Linux/macOS (zsh/bash):
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

#### For Windows (PowerShell):
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

### 4. Environment Variables
Create a `.env` file in the project root:
```bash
# Copy example environment file (if it exists)
cp .env.example .env

# Or create manually
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 5. Download Data (Optional)
The repository includes sample data files. For custom data:
```bash
# Add your own prompts to data/raw/
# Follow the JSONL format in existing files
```

## Usage Examples

### Generate Preference Pairs
```bash
python -m src.mini_cai.preference_gen \
    --prompt_file data/raw/red_team_prompts_100.jsonl \
    --sl_cai_path models/stage_02_sl_cai \
    --out_file data/processed/preferences.jsonl \
    --judge gpt-4o \
    --n_per_prompt 2
```

### Generate Harmful Response Pairs
```bash
python -m src.mini_cai.scripts.generate_harmful_pairs \
    --prompt_file data/raw/red_team_prompts_100.jsonl \
    --out_file data/raw/harmful_pairs.jsonl \
    --debug_n 2
```

### Train DPO Model
```bash
python -m src.mini_cai.train_dpo \
    --preference_file data/processed/preferences.jsonl \
    --model_path models/stage_02_sl_cai \
    --output_dir models/stage_03_dpo_cai
```

### Run Evaluation
```bash
python -m src.mini_cai.eval \
    --model_path models/stage_03_dpo_cai \
    --test_file data/raw/red_team_prompts_100.jsonl
```

## Development

### Code Formatting
```bash
# Format code with black
python -m black src/

# Sort imports with isort  
python -m isort src/

# Run tests
python -m pytest tests/
```

### Deactivating Virtual Environment
```bash
# When done working
deactivate
```

## Project Structure
```
mini-cai-dpo/
├── src/
│   └── mini_cai/           # Main package
│       ├── scripts/        # Utility scripts
│       └── prompts/        # Prompt templates
├── data/
│   ├── raw/               # Raw datasets
│   └── processed/         # Processed data
├── models/                # Model checkpoints
├── tests/                 # Test files
└── requirements.txt       # Python dependencies
```