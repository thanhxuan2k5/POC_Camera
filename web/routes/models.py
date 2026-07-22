import logging
import os
import shutil
import time
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from web.database import get_db, Model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])

class ModelResponse(BaseModel):
    id: int
    name: str
    type: str
    file_path: str
    format: str
    is_active: bool
    metadata_json: Optional[str]
    created_at: datetime
    class Config:
        from_attributes = True

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

@router.get("/", response_model=List[ModelResponse])
def list_models(db: Session = Depends(get_db)):
    return db.query(Model).all()

@router.post("/upload", response_model=ModelResponse)
async def upload_model(name: str = Form(...), type: str = Form(...), format: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_path = os.path.join(MODELS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    db_model = Model(name=name, type=type, file_path=file_path, format=format, is_active=False)
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

@router.post("/{model_id}/activate")
def activate_model(model_id: int, db: Session = Depends(get_db)):
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    db.query(Model).filter(Model.type == model.type).update({"is_active": False})
    model.is_active = True
    db.commit()
    return {"message": "Model activated"}

@router.delete("/{model_id}")
def delete_model(model_id: int, db: Session = Depends(get_db)):
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if os.path.exists(model.file_path):
        if os.path.isdir(model.file_path):
            shutil.rmtree(model.file_path)
        else:
            os.remove(model.file_path)
    db.delete(model)
    db.commit()
    return {"message": "Deleted"}

@router.get("/{model_id}/download")
def download_model(model_id: int, db: Session = Depends(get_db)):
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model or not os.path.exists(model.file_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    return FileResponse(model.file_path, filename=os.path.basename(model.file_path))

def export_task(model_id: int):
    logger.info(f"Exporting model {model_id} to TensorRT...")
    time.sleep(5)
    logger.info("Export finished")

@router.post("/export")
def export_model(model_id: int = Form(...), background_tasks: BackgroundTasks = None, db: Session = Depends(get_db)):
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if background_tasks:
        background_tasks.add_task(export_task, model_id)
    return {"message": "Export task started"}
