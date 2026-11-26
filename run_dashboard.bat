@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
REM =============================================================================
REM Proje 4: Kredi Karti Dolandiricilik Analizi - 5 Sayfali Sunum Arayuzu
REM =============================================================================

echo === Sunum Arayuzu ve Canli API Baslatiliyor ===
python app.py
pause
