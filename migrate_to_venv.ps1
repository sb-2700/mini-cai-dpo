# migrate_to_venv.ps1 - Migrate from Poetry to Virtual Environment (Windows)

Write-Host "🔄 Migrating mini-CAI-DPO from Poetry to Virtual Environment..." -ForegroundColor Blue

# Check if we're in the right directory
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "❌ Error: Not in project root directory. Please run from mini-cai-dpo/" -ForegroundColor Red
    exit 1
}

# Check Python version
$pythonVersion = python --version 2>&1 | Select-String -Pattern "\d+\.\d+" | ForEach-Object { $_.Matches.Value }
if ([version]$pythonVersion -lt [version]"3.11") {
    Write-Host "❌ Error: Python 3.11+ required. Current version: $pythonVersion" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Python version: $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "⚠️  Virtual environment already exists. Removing old one..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv
}

python -m venv venv

# Activate virtual environment
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "⬆️  Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "📥 Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

# Verify installation
Write-Host "🔍 Verifying installation..." -ForegroundColor Yellow
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')"
python -c "import tqdm; print(f'✅ tqdm: {tqdm.__version__}')"

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "📝 Creating .env file..." -ForegroundColor Yellow
    @"
# OpenAI API Key for GPT-4o judge
OPENAI_API_KEY=your_openai_api_key_here

# Add other environment variables as needed
"@ | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "⚠️  Please edit .env file with your actual API keys" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Migration completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Activate the virtual environment:" -ForegroundColor White
Write-Host "   venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Edit .env file with your API keys:" -ForegroundColor White
Write-Host "   notepad .env" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Test the setup:" -ForegroundColor White
Write-Host "   python -c `"import src.mini_cai.constants; print('✅ Package imports work')`"" -ForegroundColor Gray
Write-Host ""
Write-Host "4. To deactivate when done:" -ForegroundColor White
Write-Host "   deactivate" -ForegroundColor Gray
Write-Host ""
