@echo off
REM Neo4j Backend Setup Script for Windows

echo 🚀 Setting up Neo4j Backend for RAG File Scanner...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python detected

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Create .env file if it doesn't exist
if not exist .env (
    echo ⚙️ Creating .env file...
    copy .env.example .env
    echo 📝 Please edit .env file with your Neo4j credentials
)

echo ✅ Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file with your Neo4j credentials
echo 2. Start Neo4j database
echo 3. Run: python main.py
echo 4. Visit: http://localhost:8000/docs

pause
