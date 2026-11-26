@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
REM =============================================================================
REM Proje 4: Kredi Karti Dolandiricilik Analizi - Otomasyon Betigi (Windows CMD/BAT)
REM =============================================================================

echo === 1. Hizli Dogrulama ve Rapor Uretimi (Quick Run - 30,000 Islem) ===
python main.py --quick --sampling none

echo === 2. SMOTE Dengeleme Dogrulamasi (Quick Run + SMOTE) ===
python main.py --quick --sampling smote

echo === Tum Deneyler ve Raporlar Basariyla Tamamlandi! ===
pause
