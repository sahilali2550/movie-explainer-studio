from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import BASE_DIR, OUTPUTS_DIR, THUMBNAILS_DIR, TEMP_DIR
from app.api.v1.explainer import router as explainer_router
from app.api.v1.youtube_api import router as youtube_router

app = FastAPI(
    title="AutoExplainer AI SaaS API",
    description="Full-Auto AI Movie & Drama Explainer Video Engine with 15+ Languages, Hybrid Thumbnails, and Viral Social Suites.",
    version="1.0.0"
)

# CORS configuration for Frontend Dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for generated media
app.mount("/outputs/thumbnails", StaticFiles(directory=str(THUMBNAILS_DIR)), name="thumbnails")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Frontend Directory Mounting
FRONTEND_DIR = BASE_DIR.parent / "frontend"
if (FRONTEND_DIR / "css").exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
if (FRONTEND_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

# Include API Routers
app.include_router(explainer_router, prefix="/api/v1")
app.include_router(youtube_router, prefix="/api/v1")

@app.get("/")
def serve_frontend():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        from fastapi.responses import FileResponse
        return FileResponse(index_file)
    return {
        "status": "online",
        "service": "AutoExplainer AI SaaS Engine",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
