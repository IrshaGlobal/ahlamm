"""
Combined server for FastAPI backend and static frontend.
Serves the API on /api/* and static files from /frontend.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# Create main app first
app = FastAPI(title="Ahlamm - Full Application")

# Import and mount API
from api.main import app as api_app, load_models
app.mount("/api", api_app)

# Mount static files (frontend)
frontend_dir = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# Root endpoint serves the frontend
@app.get("/")
async def read_index():
    index_path = frontend_dir / "index.html"
    return FileResponse(str(index_path))

# Startup event to load models
@app.on_event("startup")
async def startup_event():
    print("🔄 Loading ensemble models...")
    await load_models()
    print("✅ Models loaded successfully")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Ahlamm Application")
    print("📍 Frontend: http://localhost:8000")
    print("📍 API Docs: http://localhost:8000/api/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
