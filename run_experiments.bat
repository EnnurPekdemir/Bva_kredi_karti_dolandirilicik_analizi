@echo off
REM =============================================================================
REM Proje 4: Kredi Karti Dolandiricilik Analizi - Otomasyon Betigi (Windows CMD/BAT)
REM =============================================================================

echo === Gerekli Paketlerin Kurulumu Denetleniyor ===
pip install -r requirements.txt

echo === 1. Hizli Dogrulama Testi (Quick Smoke Run) ===
python main.py --quick --sampling none

echo === 2. Tam Olcekli Boru Hatti (Full Scale Run) ===
python main.py --samples 150000 --sampling smote

echo === Tum Deneyler ve Raporlar Basariyla Tamamlandi! ===
pause
