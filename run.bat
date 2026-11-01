@echo off
title Moneyball Analytics Platform - Oakland A's Sabermetrics · by daivasukrti9.
echo =====================================================================
echo    MONEYBALL ANALYTICS PLATFORM - SABERMETRIC OPERATIONS TERMINAL
echo                         by daivasukrti9.
echo =====================================================================
echo.
echo [1/2] Verifying execution environment...

if exist ".venv\Scripts\streamlit.exe" (
    echo [2/2] Launching Streamlit application via virtual environment...
    echo Opening browser at http://localhost:8501 ...
    echo.
    .venv\Scripts\streamlit.exe run src\app\main.py
) else (
    echo [2/2] Launching via uv...
    uv run streamlit run src\app\main.py
)

pause
