"""
SatQuery AI — One-Command System Launcher
Starts the backend server and opens the interactive workstation in your web browser.
"""

import sys
import webbrowser
import time
from pathlib import Path
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("[*] SATQUERY AI - MULTIMODAL REMOTE SENSING ASSISTANT")
    print("    Problem Statement: SIH26167 | ISRO / SAC")
    print("=" * 65)
    print("[*] Initializing Agentic Orchestration Engine...")
    print("[*] Specialists Online: VLM, Grounding Head, Siamese Change Detector")
    print("[+] Workstation Interface: http://localhost:8000")
    print("=" * 65 + "\n")

    # Launch browser after a brief delay
    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open("http://localhost:8000")
        except Exception:
            pass

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Start FastAPI server
    from backend.server import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
