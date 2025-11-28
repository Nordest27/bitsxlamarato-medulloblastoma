@echo off
REM setup_windows.bat - Windows setup script using pip and venv

echo 🚀 Setting up Medulloblastoma G3/G4 Classification Environment (Windows)
echo ==================================================================

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Found Python:
python --version

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Verify installation
echo 🔍 Verifying installation...
python -c "import medulloblastoma; from medulloblastoma.dataset import download_data; from medulloblastoma.features import load_data; from medulloblastoma.plots import plot_umap_binary; print('✅ All modules imported successfully!')"

echo.
echo 🎉 Setup completed successfully!
echo.
echo To activate the environment in future sessions:
echo   venv\Scripts\activate.bat
echo.
echo To get started:
echo   jupyter lab notebooks/medulloblastoma-analysis.ipynb
echo.
echo For more information, see README.md
pause
