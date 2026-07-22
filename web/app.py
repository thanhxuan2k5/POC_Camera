import logging
import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from web.database import init_db
from web.routes import cameras, training, models, inference, inference_file, events, ws

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title='Wood Token Inspection System', version='1.0.0')

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI application...")
    # Store the main event loop in the app state for thread-safe calls
    app.state.main_event_loop = asyncio.get_running_loop()
    init_db()
    app.state.pipeline = None

# Static & Media
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(BASE_DIR, "web", "static")
EVENTS_DIR = os.path.join(BASE_DIR, "data", "events")
EXPORTS_DIR = os.path.join(BASE_DIR, "data", "exports")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(EVENTS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/storage/events", StaticFiles(directory=EVENTS_DIR), name="events")
app.mount("/storage/exports", StaticFiles(directory=EXPORTS_DIR), name="exports")

# Routers
app.include_router(cameras.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(inference.router)
app.include_router(inference_file.router)
app.include_router(events.router)
app.include_router(ws.router)

@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Wood Token Inspection API - Please place index.html in web/static/"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
