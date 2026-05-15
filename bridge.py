"""
bridge.py — Local ↔ Cloud MCP Bridge

Connects Claude Desktop (via stdio) to the remote MCP trading server
hosted on Google Cloud Run. Runs two threads:
  - listen_to_server  : maintains an SSE connection and forwards server
                        messages to Claude via stdout.
  - listen_to_cloud   : reads Claude's JSON-RPC messages from stdin and
                        POSTs them to the server.

Configuration is via environment variables (see .env.example).
"""

import sys
import os
import requests
import threading
import json
import time
import datetime

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)

# ──────────────────────────────────────────────
#  Configuration  (loaded from environment)
# ──────────────────────────────────────────────
API_KEY  = os.environ["MCP_API_KEY"]
BASE_URL = os.getenv("MCP_SERVER_URL", "https://mcp-market-data-356995198599.us-east1.run.app")
LOG_DIR  = os.getenv("LOG_DIR", os.path.dirname(os.path.abspath(__file__)))

SSE_URL  = f"{BASE_URL}/sse"
LOG_FILE = os.path.join(LOG_DIR, "bridge_debug_log.txt")

# ──────────────────────────────────────────────
#  Global State
# ──────────────────────────────────────────────
post_url       = None
is_initialized = False


# ──────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────
def log(msg: str) -> None:
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted)
    except OSError:
        pass
    try:
        sys.stderr.write(formatted)
    except OSError:
        pass


# ──────────────────────────────────────────────
#  SSE Listener  (server → Claude)
# ──────────────────────────────────────────────
def listen_to_server() -> None:
    """Maintains a persistent SSE connection to the cloud server.
    Forwards JSON-RPC responses to Claude via stdout.
    Auto-reconnects on failure.
    """
    global post_url, is_initialized

    while True:
        log("--- ATTEMPTING SSE CONNECTION ---")
        try:
            headers = {"Accept": "text/event-stream", "x-api-key": API_KEY}
            with requests.get(SSE_URL, stream=True, headers=headers, timeout=None) as r:
                log(f"Stream Status: {r.status_code}")
                if r.status_code != 200:
                    log(f"Server returned {r.status_code}, retrying in 5s...")
                    time.sleep(5)
                    continue

                for line in r.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line

                    if not decoded.startswith("data: "):
                        continue

                    data = decoded[6:].strip()

                    if data.startswith("/messages"):
                        post_url = f"{BASE_URL}{data}"
                        is_initialized = False
                        log(f"New Session: {post_url}")
                    elif data.startswith("http"):
                        post_url = data
                        is_initialized = False
                        log(f"Full Post URL: {post_url}")
                    else:
                        # Forward server response to Claude
                        try:
                            sys.stdout.write(data + "\n")
                            sys.stdout.flush()
                        except Exception as e:
                            log(f"Stdout write error: {e}")

        except Exception as e:
            log(f"SSE Connection Lost: {e}")
            post_url = None
            is_initialized = False

        time.sleep(2)


# ──────────────────────────────────────────────
#  Stdin Listener  (Claude → server)
# ──────────────────────────────────────────────
def listen_to_cloud() -> None:
    """Reads JSON-RPC messages from Claude on stdin and POSTs them to the server."""
    global post_url, is_initialized

    for line in sys.stdin:
        msg = line.strip()
        if not msg:
            continue

        # Wait for SSE session to be established
        while not post_url:
            time.sleep(0.1)

        try:
            msg_json = json.loads(msg)
            method = msg_json.get("method", "unknown")

            if method != "initialize" and not is_initialized:
                log(f"Waiting for session to stabilise before sending '{method}'...")
                time.sleep(4)
                is_initialized = True

            log(f"Claude → Server [{method}]")

            resp = requests.post(
                post_url,
                json=msg_json,
                timeout=120,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": API_KEY,
                },
            )

            if method == "initialize":
                log("Initialisation sent. Buffering next requests...")
                time.sleep(4.0)
                is_initialized = True

            log(f"Server Status: {resp.status_code}")

        except Exception as e:
            log(f"POST Error: {e}")
            post_url = None
            is_initialized = False


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"Bridge Log Init — {datetime.datetime.now()}\n")

    threading.Thread(target=listen_to_server, daemon=True).start()
    listen_to_cloud()