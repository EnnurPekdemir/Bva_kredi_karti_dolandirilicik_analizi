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
    Hem statik dosyaları (HTML/PNG/JS) sunar hem de canlı pipeline çalıştırma
    isteğini (/api/run-pipeline) UTF-8 güvenliğiyle yönetir.
    """
    def do_POST(self):
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
