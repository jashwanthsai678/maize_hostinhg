# 🏁 Quick-Start: Demo Readiness Guide

This document covers everything you need to do **before** and **during** your presentation to ensure the system runs flawlessly.

## 📋 1. Pre-Run Checklist (Do these early)

### A. Local Machine Setup
Ensure you have the required Python libraries installed:
```powershell
pip install fastapi uvicorn ultralytics pillow requests httpx python-multipart python-dotenv joblib pandas numpy
```

### B. Remote VM Setup (The "AI Advisor")
Your VM handles the most complex logic. Ensure:
1.  **IP Address**: Verify the IP is still `34.69.210.248`.
2.  **Firewall**: Port **8000** must be open in your cloud firewall (GCP/AWS).
3.  **Active Service**: Uvicorn must be running on the VM:
    ```bash
    uvicorn main_advisory:app --host 0.0.0.0 --port 8000
    ```

### C. Configuration File (`.env`)
Check the `.env` file in your local folder. It should look like this:
```env
LLM_API_URL=http://34.69.210.248:8000/advisory
YOLO_API_URL=http://127.0.0.1:8002/predict
YIELD_API_URL=http://127.0.0.1:8003/predict
```

---

## 🚀 2. Launch Commands (The Boring Way)

If you want to show the terminal logs to your boss, open **3 separate windows** and run:

1.  **Terminal 1 (YOLO)**: `uvicorn yolo_api:app --host 127.0.0.1 --port 8002`
2.  **Terminal 2 (Yield)**: `python app_blended.py`
3.  **Terminal 3 (Orchestrator)**: `uvicorn orchestrator_api:app --host 127.0.0.1 --port 8001`

---

## ⚡ 3. The "One-Click" Demo (The Professional Way)

Just double-click the **`run_demo.bat`** file in your folder. 
*   It handles all the terminals for you.
*   It automatically opens your browser.
*   It ensures the services start in the correct order.

---

## 🧪 4. How to Verify Everything is Alive
Before your boss arrives, open these links in your browser. If you see JSON text, the service is working!

*   ✅ **YOLO**: [http://127.0.0.1:8002/](http://127.0.0.1:8002/)
*   ✅ **Yield**: [http://127.0.0.1:8003/](http://127.0.0.1:8003/)
*   ✅ **Website**: [http://127.0.0.1:8001/](http://127.0.0.1:8001/) (YOUR MAIN DASHBOARD)

---

## 💡 Demo Performance Tip
Generative AI can take time. If the **"Agronomist Advisory"** takes 30-60 seconds to appear, explain to your boss: 
*"The system is currently fusing the visual diagnosis with local environmental data and performing deep-reasoning on our cloud VM to generate tailored advice."*
