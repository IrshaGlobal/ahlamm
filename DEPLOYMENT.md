# 🚀 Deployment Guide (FastAPI + Frontend)

This app runs a FastAPI backend and a modern HTML/CSS/JS frontend via a single Python server (`server.py`).

If you're deploying now, follow the FastAPI guide here:

- See `DEPLOYMENT_FASTAPI.md` for detailed, up-to-date steps and options.

Below is a lightweight summary of the best options.

---

## Quick Options

### 1) Hugging Face Spaces (Docker) – Free

1. Create a Space with SDK = Docker
2. Push this repo (it already contains a working Dockerfile)
3. Ensure port is correct (HF default is 7860). If needed, change the container port in the Dockerfile and `server.py` run command.

### 2) Railway (Docker) – Simple

1. Connect GitHub repo
2. Railway auto-detects Dockerfile and builds the image
3. Exposes port 8000 by default

### 3) Render (Docker) – Free tier

1. New → Web Service → Docker
2. Use `./Dockerfile`
3. Instance type: Free or Starter (2GB RAM recommended for TensorFlow)

### 4) Google Cloud Run – Serverless

1. Build image with Cloud Build
2. Deploy to Cloud Run with `--port 8000` and `--memory 2Gi`

---

## Docker (Local or Any Cloud)

Build and run:

```bash
docker build -t ahlamm-app .
docker run -p 8000:8000 ahlamm-app
```

Then open http://localhost:8000

---

## Notes and Tips

- Models are included in `model/` and loaded at startup; allow ~10-15s cold start on small instances.
- Health endpoint: `GET /api/health` (used by the container healthcheck).
- For split deployment (separate frontend domain), set stricter CORS in `api/main.py`.

For the full instructions and troubleshooting, read `DEPLOYMENT_FASTAPI.md`.
