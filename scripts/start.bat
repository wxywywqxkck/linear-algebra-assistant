@echo off
chcp 65001 >nul
echo.
echo ================================
echo  Linear Algebra RAG Assistant
echo ================================
echo.

REM Always run from the project root dir, wherever this .bat is called from
cd /d "%~dp0.."
set "PROJECT_ROOT=%cd%"

REM Skip Streamlit first-run onboarding email prompt
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_BROWSER_SHOW_ERROR_DETAILS=false

REM Check .env file
if not exist ".env" (
    echo [WARN] .env not found. Copy .env.example to .env and fill in OPENAI_API_KEY
    echo        copy .env.example .env
    echo.
)

REM Pick Python
where python >nul 2>nul
if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
    echo [INFO] Using venv
) else (
    set "PYTHON=python"
    echo [INFO] Using system Python
)

echo [INFO] Project root: %PROJECT_ROOT%
echo.

echo [1/2] Starting FastAPI backend on port 8000...
REM /d sets working dir for the new window so "import src" works
start "FastAPI Backend" /d "%PROJECT_ROOT%" cmd /k "python -m uvicorn src.api:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak >nul

echo [2/2] Starting Streamlit frontend on port 8501...
echo.
echo ========================================
echo  Frontend: http://localhost:8501
echo  API Docs: http://localhost:8000/docs
echo ========================================
echo.

"%PYTHON%" -m streamlit run src/app.py --server.port 8501 --server.headless true
