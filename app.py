"""
app.py
=============================================================================
Proje 4: Kredi Karti Dolandiricilik Analizi - Web Sunum Arayuzu & Live API
=============================================================================
Sıfır kurulum ve ek paket gerektirmeden çoklu iş parçacıklı (ThreadingHTTPServer)
yerel HTTP sunucusunu başlatır, web arayüzünden tek tıkla arka plandaki
Python/PySpark pipeline'ını canlı çalıştırma API'sini sunar.
"""

import os
import sys
import json
import subprocess
import webbrowser
import http.server
import socket

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def find_available_port(start_port=8080):
    port = start_port
    while port < 8150:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    return 8080


class LivePresentationHandler(http.server.SimpleHTTPRequestHandler):
    """
    Hem statik dosyaları (HTML/PNG/JS) önbelleksiz (no-cache) sunar hem de
    canlı pipeline çalıştırma isteğini (/api/run-pipeline) UTF-8 güvenliğiyle yönetir.
    """
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_POST(self):
        if self.path == '/api/predict-transaction':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8', errors='replace') if content_length > 0 else "{}"
            try:
                payload = json.loads(body)
                amount = float(payload.get("amount", 180.0))
                cust_avg = max(10.0, float(payload.get("cust_avg", 150.0)))
                hour = int(payload.get("hour", 14))
                tx_1h = int(payload.get("tx_1h", 1))
                delta_sec = float(payload.get("delta_sec", 18000.0))
                pca_shift = float(payload.get("pca_shift", 0.0))

                # PySpark Davranışsal Özelliklerin Canlı Hesaplanması (Öğrenci 3)
                cust_std = max(25.0, cust_avg * 0.45)
                deviation = amount - cust_avg
                zscore = deviation / cust_std

                # XGBoost / GBDT Karar Skoru (Log-Odds Katkı Analizi)
                logit = -4.2
                reasons = []

                # 1. Tutar Z-Skoru Katkısı
                if zscore > 2.0:
                    z_contrib = min(3.2, (zscore - 1.5) * 0.45)
                    logit += z_contrib
                    reasons.append(f"PySpark Tutar Z-Skoru yüksek (+{zscore:.1f}σ sapma)")
                else:
                    reasons.append(f"Tutar müşteri ortalamasıyla uyumlu (Z={zscore:.1f}σ)")

                # 2. Son 1 Saatteki İşlem Sıklığı (Velocity Attack)
                if tx_1h >= 3:
                    v_contrib = min(3.0, (tx_1h - 2) * 0.65)
                    logit += v_contrib
                    reasons.append(f"PySpark Rolling Window: Son 1 saatte {tx_1h} ardışık işlem (Velocity Attack)")

                # 3. İşlemler Arası Geçen Süre (Delta Time)
                if delta_sec < 120:
                    logit += 1.45
                    reasons.append(f"PySpark Delta Time çok kısa ({delta_sec:.0f} sn - Bot/Hızlı Çekim şüphesi)")
                elif delta_sec < 600:
                    logit += 0.65

                # 4. Gece Saati Etkisi (Öğrenci 1 EDA Bulgusu)
                if 0 <= hour <= 6:
                    logit += 1.15
                    reasons.append(f"EDA Zaman Bulgusu: Gece saat diliminde işlem ({hour:02d}:00)")

                # 5. PCA (V14, V12, V4) Anomali Sapması
                if pca_shift > 0.5:
                    logit += pca_shift * 1.1
                    reasons.append(f"PCA Bileşen Anomalisi (V14/V12/V4 sapma skoru: {pca_shift:.1f})")

                import math
                prob = 1.0 / (1.0 + math.exp(-logit))

                if prob >= 0.75:
                    decision = "FRAUD (DOLANDIRICILIK) - KART BLOKE EDİLDİ"
                    level = "critical"
                elif prob >= 0.35:
                    decision = "ŞÜPHELİ İŞLEM - 3D SECURE / SMS ONAYI GÖNDERİLDİ"
                    level = "warning"
                else:
                    decision = "NORMAL İŞLEM - ONAYLANDI"
                    level = "safe"

                res = {
                    "status": "success",
                    "probability": round(prob * 100, 2),
                    "zscore": round(zscore, 2),
                    "deviation": round(deviation, 2),
                    "decision": decision,
                    "level": level,
                    "reasons": reasons
                }
            except Exception as e:
                res = {"status": "error", "message": str(e)}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode('utf-8'))
            return

        if self.path == '/api/run-pipeline':
            print("\n[LIVE DEMO] Web arayuzunden canli pipeline calistirma tetiklendi...")
            try:
                child_env = os.environ.copy()
                child_env["PYTHONIOENCODING"] = "utf-8"
                child_env["PYTHONUTF8"] = "1"

                process = subprocess.run(
                    [sys.executable, "main.py", "--quick"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=child_env
                )
                output = ""
                if process.stderr:
                    output += process.stderr + "\n"
                if process.stdout:
                    output += process.stdout

                response_data = {
                    "status": "success" if process.returncode == 0 else "error",
                    "code": process.returncode,
                    "output": output
                }
            except Exception as e:
                response_data = {
                    "status": "error",
                    "code": 1,
                    "output": f"Hata olustu: {str(e)}"
                }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_error(404, "Endpoint bulunamadi")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


def start_server():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    port = find_available_port(8080)

    server_cls = getattr(http.server, "ThreadingHTTPServer", http.server.HTTPServer)
    server_cls.allow_reuse_address = True
    with server_cls(("127.0.0.1", port), LivePresentationHandler) as httpd:
        url = f"http://localhost:{port}/index.html"
        print("\n" + "="*80, flush=True)
        print("PROJE 4: KREDI KARTI DOLANDIRICILIK ANALIZI - CANLI SUNUM PORTALI", flush=True)
        print("="*80, flush=True)
        print(f"Sunucu Adresi: {url}", flush=True)
        print("Web arayuzu ve Canli API devrede...", flush=True)
        print("Durdurmak icin terminalde Ctrl+C tuslarina basabilirsiniz.\n", flush=True)

        try:
            webbrowser.open(url)
        except Exception:
            pass

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nSunucu guvenle kapatildi. Iyi sunumlar!", flush=True)


if __name__ == "__main__":
    start_server()
