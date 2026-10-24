#!/usr/bin/env python3
"""
V4 DASHBOARD TUNNEL — Uses pyngrok for global access
"""
from pyngrok import ngrok
import time
import sys

# Set your ngrok auth token (free signup at ngrok.com)
# NGROK_AUTH_TOKEN = "your_token_here"

print("=" * 50)
print("V4 TRADING DASHBOARD — Tunnel Setup")
print("=" * 50)

# Check if streamlit is running
import urllib.request
try:
    urllib.request.urlopen("http://localhost:8501", timeout=5)
    print("✅ Streamlit is running on port 8501")
except:
    print("❌ Streamlit is NOT running!")
    print("   Start it first with: python -m streamlit run dashboard.py")
    sys.exit(1)

# Start ngrok tunnel
print("\n🚀 Starting ngrok tunnel...")

# If you have a token, uncomment:
# ngrok.set_auth_token(NGROK_AUTH_TOKEN)

tunnel = ngrok.connect(8501, "http")

print("\n" + "=" * 50)
print("🌍 YOUR GLOBAL URL:")
print(f"   {tunnel.public_url}")
print("=" * 50)
print("\nShare this URL with anyone — works on mobile!")
print("Ctrl+C to stop.")

try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("\n👋 Stopping tunnel...")
    ngrok.kill()
    print("Done.")
