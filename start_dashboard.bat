@echo off
REM V4 Trading Dashboard Launcher
cd /d C:\Users\david_wncs0ps\jarvis\trading
python -m streamlit run dashboard.py --server.port 8501 --server.headless true
