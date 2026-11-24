"""
app.py
=============================================================================
Proje 4: Kredi Karti Dolandiricilik Analizi - Web Sunum Arayuzu & Live API
=============================================================================
Sıfır kurulum ve ek paket gerektirmeden yerel HTTP sunucusunu başlatır,
web arayüzünden tek tıkla arka plandaki Python/PySpark pipeline'ını canlı
çalıştırma API'sini sunar ve tarayıcıyı otomatik açar.
"""

import os
import sys
import json
import subprocess
import webbrowser
import http.server
import socketserver
import socket

def find_available_port(start_port=8080):
    port = start_port
    while port < 9000:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
            port += 1
    return 8080

class LivePresentationHandler(http.server.SimpleHTTPRequestHandler):
    """
    Hem statik dosyaları (HTML/PNG/JS) sunar hem de canlı pipeline çalıştırma
    isteğini (/api/run-pipeline) yönetir.
    """
    def do_POST(self):
        if self.path == '/api/run-pipeline':
            print("\n[LIVE DEMO] Web arayüzünden canlı pipeline çalıştırma tetiklendi...")
            try:
                # python main.py --quick komutunu çalıştır
                process = subprocess.run(
                    [sys.executable, "main.py", "--quick"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace"
                )
                output = process.stdout
                if process.stderr:
                    output += "\n" + process.stderr

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
    
    print("\n" + "="*80)
    print("PROJE 4: KREDİ KARTI DOLANDIRICILIK ANALİZİ - CANLI SUNUM PORTALI")
    print("="*80)
    print(f"Sunucu Adresi: http://localhost:{port}/index.html")
    print("Web arayüzü ve Canlı API devrede...")
    print("Durdurmak için terminalde Ctrl+C tuşlarına basabilirsiniz.\n")

    webbrowser.open(f"http://localhost:{port}/index.html")

    # Reuse address to prevent 'Address already in use' errors
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), LivePresentationHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nSunucu güvenle kapatıldı. İyi sunumlar!")

if __name__ == "__main__":
    start_server()
