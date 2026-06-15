



```bat
@echo off
cd /d %~dp0
call conda activate smart_accident_streamlitapp
streamlit run streamlit_app.py
pause
