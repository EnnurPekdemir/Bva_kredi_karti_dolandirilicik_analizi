"""
app.py
=============================================================================
Proje 4: Kredi Karti Dolandiricilik Analizi - Web Sunum Arayuzu Sunucusu
=============================================================================
Sıfır kurulum ve ek paket gerektirmeden yerel HTTP sunucusunu başlatır
ve 5 sayfalı interaktif sunum portalını otomatik olarak tarayıcıda açar.
"""

import os
import sys
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

def start_server():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    port = find_available_port(8080)
    handler = http.server.SimpleHTTPRequestHandler
    
    print("\n" + "="*80)
    print("PROJE 4: KREDİ KARTI DOLANDIRICILIK ANALİZİ - SUNUM ARAYÜZÜ BAŞLATILIYOR")
    print("="*80)
    print(f"\nSunucu Adresi: http://localhost:{port}/index.html")
    print("Tarayıcınız otomatik açılıyor... (Durdurmak için Ctrl+C tuşlarına basabilirsiniz)\n")

    webbrowser.open(f"http://localhost:{port}/index.html")

    with socketserver.TCPServer(("", port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nSunucu kapatıldı. İyi sunumlar!")

if __name__ == "__main__":
    start_server()
