@echo off
echo Starting Mini AutoML Backend...
cd /d %~dp0
call venv\Scripts\activate
uvicorn main:app --reload --port 8000
