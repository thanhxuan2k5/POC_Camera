import logging
import cv2
import base64
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel
from web.database import get_db, Camera

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cameras", tags=["cameras"])

class CameraCreate(BaseModel):
    name: str
    rtsp_url: str
    is_active: bool = True

class CameraResponse(CameraCreate):
    id: int
    class Config:
        from_attributes = True

@router.get("/", response_model=List[CameraResponse])
def list_cameras(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(Camera).offset(skip).limit(limit).all()

@router.post("/", response_model=CameraResponse)
def create_camera(cam: CameraCreate, db: Session = Depends(get_db)):
    try:
        db_cam = Camera(**cam.dict())
        db.add(db_cam)
        db.commit()
        db.refresh(db_cam)
        return db_cam
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error while creating camera: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while creating camera: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(camera_id: int, db: Session = Depends(get_db)):
    db_cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return db_cam

@router.put("/{camera_id}", response_model=CameraResponse)
def update_camera(camera_id: int, cam: CameraCreate, db: Session = Depends(get_db)):
    db_cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    for k, v in cam.dict().items():
        setattr(db_cam, k, v)
    db.commit()
    db.refresh(db_cam)
    return db_cam

@router.delete("/{camera_id}")
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    db_cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    db.delete(db_cam)
    db.commit()
    return {"message": "Deleted"}

@router.post("/{camera_id}/test")
def test_camera(camera_id: int, db: Session = Depends(get_db)):
    db_cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    cap = cv2.VideoCapture(db_cam.rtsp_url)
    if not cap.isOpened():
        return {"success": False, "message": "Failed to open RTSP stream"}
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {"success": False, "message": "Failed to read frame"}
    
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return {"success": True, "message": "Connection successful", "snapshot": f"data:image/jpeg;base64,{jpg_as_text}"}

@router.get("/{camera_id}/snapshot")
def get_snapshot(camera_id: int, db: Session = Depends(get_db)):
    db_cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    cap = cv2.VideoCapture(db_cam.rtsp_url)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open camera stream")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(status_code=500, detail="Could not read frame")
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return {"snapshot": f"data:image/jpeg;base64,{jpg_as_text}"}
