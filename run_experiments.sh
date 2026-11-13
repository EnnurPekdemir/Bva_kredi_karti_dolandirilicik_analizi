#!/usr/bin/env bash
# =============================================================================
# Proje 4: Kredi Karti Dolandiricilik Analizi - Otomasyon Betigi (Bash)
# =============================================================================

set -e

echo "=== Gerekli Paketlerin Kurulumu Denetleniyor ==="
pip install -r requirements.txt

echo "=== 1. Hizli Dogrulama Testi (Quick Smoke Run) ==="
python main.py --quick --sampling none

echo "=== 2. Tam Olcekli Boru Hatti (Full Scale Run - SMOTE Dengeleme) ==="
python main.py --samples 150000 --sampling smote

echo "=== 3. Tum Deneyler ve Raporlar Basariyla Tamamlandi! ==="
