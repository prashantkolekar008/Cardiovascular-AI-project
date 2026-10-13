@echo off
echo Starting Cardiovascular AI Backend Service...
cd /d "%~dp0"
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
pause
