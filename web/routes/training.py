"""Training API — upload reference images, extract embeddings, build classifier."""

import logging
import os
import glob
import shutil
import numpy as np
import cv2
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session

from web.database import get_db, TrainingSession, ReferenceEmbedding, SessionLocal, Model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SessionResponse(BaseModel):
    id: int
    status: str
    num_images: int
    threshold: float
    accuracy: float
    log_text: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Background training task for Anomaly Detection
# ---------------------------------------------------------------------------

def _run_training_anomaly(session_id: int, ref_dir: str, threshold: float = 0.85):
    """
    Background task for Anomaly Detection.
    Loads all images, treats them as the 'normal' class, and saves a single centroid.
    """
    db = SessionLocal()
    log_lines = []
    try:
        session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not session:
            return

        session.status = "running"
        session.started_at = datetime.utcnow()
        db.commit()

        from core.embedder import TokenEmbedder

        device = "cpu"
        try:
            import yaml
            with open("pipeline_config.yaml", "r") as f:
                cfg = yaml.safe_load(f) or {}
            device = cfg.get("embedding", {}).get("device", "cpu")
        except Exception:
            pass

        embedder = TokenEmbedder(device=device)
        log_lines.append(f"Embedder loaded on device={device}")

        image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        image_paths = []
        for ext in image_exts:
            image_paths.extend(glob.glob(os.path.join(ref_dir, ext)))

        if not image_paths:
            raise ValueError("No reference images found.")

        log_lines.append(f"Found {len(image_paths)} 'normal' reference images.")
        session.num_images = len(image_paths)
        db.commit()

        embeddings = []
        for i, img_path in enumerate(image_paths):
            try:
                img = cv2.imread(img_path)
                if img is None: continue
                
                # Augment by rotating the image every 15 degrees to make it rotation-invariant
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                for angle in range(0, 360, 15):
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    emb = embedder.extract(rotated)
                    if emb is not None:
                        embeddings.append(emb)
            except Exception as e:
                log_lines.append(f"  ERROR on {os.path.basename(img_path)}: {e}")

        if not embeddings:
            raise ValueError("Could not extract any valid embeddings.")

        # Save ALL embeddings as a matrix for Top-1 matching (no centroid averaging)
        emb_array = np.array(embeddings)
        
        model = session.model
        emb_dir = model.file_path if model else os.path.join(BASE_DIR, "data", "models", f"session_{session_id}")
        os.makedirs(emb_dir, exist_ok=True)

        # Save the full array of augmented embeddings
        np.save(os.path.join(emb_dir, "san_pham_tot.npy"), emb_array)
        log_lines.append(f"Saved 'san_pham_tot.npy' (matrix of {len(embeddings)} augmented embeddings for rotation invariance).")

        # --- SYNC TO PIPELINE'S EMBEDDING DIR ---
        try:
            target_emb_dir = os.path.join(BASE_DIR, "data", "embeddings")
            os.makedirs(target_emb_dir, exist_ok=True)
            log_lines.append(f"Syncing to pipeline directory: {target_emb_dir}")
            for f in glob.glob(os.path.join(target_emb_dir, '*.npy')):
                os.remove(f)
            shutil.copy(os.path.join(emb_dir, "san_pham_tot.npy"), target_emb_dir)
            log_lines.append("Synchronization complete.")
        except Exception as sync_e:
            log_lines.append(f"ERROR: Failed to sync embeddings to pipeline directory: {sync_e}")
        # --- END OF SYNC ---

        session.status = "completed"
        session.completed_at = datetime.utcnow()
        session.log_text = "\n".join(log_lines)
        db.commit()

        logger.info("Anomaly training session %d completed.", session_id)

    except Exception as e:
        logger.error("Training session %d failed: %s", session_id, e, exc_info=True)
        # ... (error handling remains the same)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_references(
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Upload reference images for the 'normal' class for anomaly detection.
    A new session is created for each upload batch.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Create a new model and session
    name = model_name or f"Model_Anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Check for duplicate name
    existing = db.query(Model).filter(Model.name == name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Model name '{name}' already exists.")
        
    model_dir = os.path.join(BASE_DIR, "data", "models", name)
    os.makedirs(model_dir, exist_ok=True)
    
    db_model = Model(name=name, type="embedder", file_path=model_dir, format="npy", is_active=False)
    db.add(db_model)
    db.commit()
    db.refresh(db_model)

    session = TrainingSession(status="pending", num_images=0, model_id=db_model.id)
    db.add(session)
    db.commit()
    db.refresh(session)

    # Define the directory for the session's reference folder
    ref_dir = os.path.join(BASE_DIR, "data", "references", f"session_{session.id}")
    os.makedirs(ref_dir, exist_ok=True)

    saved_count = 0
    for file in files:
        safe_filename = file.filename.replace("..", "").replace("/", "_").replace("\\", "_")
        file_path = os.path.join(ref_dir, safe_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_count += 1
    
    session.num_images = saved_count
    db.commit()

    return {
        "session_id": session.id,
        "images_uploaded": saved_count,
        "message": f"Uploaded {saved_count} images for anomaly detection. Call POST /api/training/start?session_id={session.id} to begin.",
    }


@router.post("/start")
def start_training(
    session_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Start embedding extraction in background."""
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Training session not found")
    if session.status == "running":
        return {"message": "Training already in progress"}

    ref_dir = os.path.join(BASE_DIR, "data", "references", f"session_{session_id}")
    if not os.path.isdir(ref_dir):
        raise HTTPException(status_code=400, detail=f"No reference images found for session {session_id}")

    # Use the new anomaly training task
    background_tasks.add_task(_run_training_anomaly, session_id, ref_dir)
    return {"message": "Anomaly detection training started", "session_id": session_id}


# Other routes (get_status, list_sessions, etc.) remain largely the same
# and do not need modification for this logic change.

@router.get("/status")
def get_training_status(db: Session = Depends(get_db)):
    running = db.query(TrainingSession).filter(TrainingSession.status == "running").all()
    return [{"id": s.id, "status": s.status, "num_images": s.num_images, "started_at": s.started_at.isoformat() if s.started_at else None} for s in running]

@router.get("/sessions")
def list_sessions(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    sessions = db.query(TrainingSession).order_by(TrainingSession.id.desc()).offset(skip).limit(limit).all()
    return [{"id": s.id, "status": s.status, "num_images": s.num_images, "threshold": s.threshold or 0.0, "accuracy": s.accuracy or 0.0, "log_text": s.log_text, "started_at": s.started_at.isoformat() if s.started_at else None, "completed_at": s.completed_at.isoformat() if s.completed_at else None} for s in sessions]

@router.get("/sessions/{session_id}")
def get_session(session_id: int, db: Session = Depends(get_db)):
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.post("/validate")
async def validate_model(files: List[UploadFile] = File(...)):
    emb_path = os.path.join(BASE_DIR, "data", "embeddings")
    if not os.path.exists(emb_path) or not os.listdir(emb_path):
        raise HTTPException(status_code=400, detail="No trained embeddings found. Train first.")

    from main_app.classifier import Classifier
    classifier = Classifier.from_config()
    
    results = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: continue

        _, score, class_name = classifier.predict(img)
        results.append({"file": file.filename, "class_name": class_name, "similarity": round(score, 4)})

    return {"total": len(results), "results": results}
