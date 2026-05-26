@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python -u app.py
pause
