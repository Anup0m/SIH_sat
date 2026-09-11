# SatQuery AI — Production Cloud & Local Deployment Guide 🛰️
**Smart India Hackathon 2026 | Problem Statement: SIH26167 (ISRO / SAC)**

---

## Option 1: Render.com (Recommended — 100% Free & 1-Click GitHub Deploy)

Render provides free hosting for web services directly connected to your GitHub repository.

### Steps:
1. Go to **[https://render.com](https://render.com)** and sign in with your GitHub account.
2. Click **"New +"** $\to$ **"Web Service"**.
3. Connect your repository: **`Anup0m/SIH_sat`**.
4. Render will automatically detect the settings from `render.yaml` or you can enter:
   - **Name**: `satquery-ai`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.server:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`
5. Click **"Create Web Service"**.
6. In 2-3 minutes, Render will assign you a live HTTPS public URL:  
   👉 `https://satquery-ai.onrender.com`

---

## Option 2: Railway.app (Fast Cloud Deploy)

1. Go to **[https://railway.app](https://railway.app)** and sign in with GitHub.
2. Click **"New Project"** $\to$ **"Deploy from GitHub repo"**.
3. Select **`Anup0m/SIH_sat`**.
4. Railway automatically detects the `Dockerfile` or `Procfile` and builds the service.
5. In Project Settings $\to$ **Networking**, click **"Generate Domain"**.
6. You instantly get a live public HTTPS address:  
   👉 `https://satquery-ai.up.railway.app`

---

## Option 3: Hugging Face Spaces (Ideal for ML / Hackathon Evaluation)

1. Go to **[https://huggingface.co/spaces](https://huggingface.co/spaces)** and click **"Create new Space"**.
2. Set Space Name: `SatQuery-AI`.
3. Choose **Docker** as the Space SDK (Blank).
4. Set Space hardware to **Free CPU (16 GB RAM)**.
5. Push this repository or connect GitHub:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/SatQuery-AI
   git push space main
   ```
6. Hugging Face builds the Docker image and hosts it with a persistent global URL.

---

## Option 4: Docker Container (Local / AWS / DigitalOcean / College Server)

To deploy on any server or VPS in a single command:

```bash
# Clone the repository
git clone https://github.com/Anup0m/SIH_sat.git
cd SIH_sat

# Build and run with Docker Compose
docker compose up -d --build
```

The application will be live at: `http://<your-server-ip>:8000`

---

## Option 5: Instant Live Public URL from Your Laptop (Zero Cloud Setup)

If you have the server running locally on your laptop (`python run_satquery.py`) and want to share a live public link right now with the jury or test on your mobile:

### Using Cloudflare Tunnel (Free, No Sign-up, Secure HTTPS):
```powershell
# In PowerShell:
./cloudflared.exe tunnel --url http://localhost:8000
```
*Gives an instant URL like: `https://random-words.trycloudflare.com`*

### Using Pinggy (Instant SSH Tunnel):
```powershell
ssh -p 443 -R0:localhost:8000 a.pinggy.io
```

---

## 🔍 Verification Endpoints

Once deployed on any platform, verify health:
- **Interactive UI**: `https://<your-deployed-domain>/`
- **Analysis Dashboard**: `https://<your-deployed-domain>/output.html`
- **System Health API**: `https://<your-deployed-domain>/api/health`
- **Model Registry API**: `https://<your-deployed-domain>/api/models`
- **Curated Dataset Download**: `https://<your-deployed-domain>/api/download_test_dataset`
