"""Inference pipeline control API — start/stop the AI inspection pipeline."""

import logging
import os
import time
import threading
import base64
import cv2
import yaml
import asyncio
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session

from web.database import get_db, Camera, Event, SessionLocal, Model
from core.pipeline import InspectionPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/inference", tags=["inference"])

class LiveStartSettings(BaseModel):
    model_id: int
    confidence_threshold: float
    similarity_threshold: float

def _save_event(track_id, result, similarity, frame, bbox, camera_id):
    # Implementation is correct, keeping it as is.
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        events_dir = os.path.join("data", "events", today)
        os.makedirs(events_dir, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S_%f")
        img_name = f"track{track_id}_{result}_{ts}.jpg"
        img_path = os.path.join(events_dir, img_name)
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad = 20
        x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
        cropped = frame[y1:y2, x1:x2].copy()
        cv2.imwrite(img_path, cropped)
        db = SessionLocal()
        try:
            event = Event(camera_id=camera_id, token_track_id=str(track_id), result=result, similarity_score=similarity, image_path=img_path)
            db.add(event)
            db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to save event: {e}", exc_info=True)

@router.post("/start/{camera_id}")
def start_inference(camera_id: int, settings: LiveStartSettings, request: Request, db: Session = Depends(get_db)):
    if getattr(request.app.state, "pipeline", None) and request.app.state.pipeline.is_running():
        raise HTTPException(status_code=400, detail="Pipeline is already running.")

    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam or not cam.is_active:
        raise HTTPException(status_code=404, detail="Active camera not found")

    model = db.query(Model).filter(Model.id == settings.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Selected model not found")

    try:
        with open("pipeline_config.yaml", "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server configuration error: {e}")

    # Build config dynamically using request settings and file fallbacks
    pipeline_config = {
        "rtsp_url": cam.rtsp_url,
        "use_gstreamer": cfg.get("camera", {}).get("use_gstreamer", False),
        "frame_skip": cfg.get("camera", {}).get("frame_skip", 2),
        "yolo_model": cfg.get("detection", {}).get("model_path"),
        "device": cfg.get("detection", {}).get("device", "cpu"),
        "max_disappeared": cfg.get("tracker", {}).get("max_disappeared", 15),
        "max_distance": cfg.get("tracker", {}).get("max_distance", 80),
        "decision_line_y": cfg.get("tracker", {}).get("decision_line_y", 0.75),
        
        # Overrides from user
        "conf_threshold": settings.confidence_threshold,
        "anomaly_threshold": settings.similarity_threshold,
        "references_dir": model.file_path, # Use the selected model's path
    }

    logger.info("Starting pipeline with dynamic config: %s", pipeline_config)
    pipeline = InspectionPipeline(pipeline_config)
    
    main_loop = request.app.state.main_event_loop
    from web.routes.ws import manager as ws_manager

    def on_result(track_id, result, similarity, frame, bbox):
        threading.Thread(target=_save_event, args=(track_id, result, similarity, frame, bbox, cam.id), daemon=True).start()
        event_data = {"type": "event", "data": {"result": result, "similarity": round(similarity, 4)}}
        
        if result == "NG":
            try:
                x1, y1, x2, y2 = bbox
                h, w = frame.shape[:2]
                pad = 20
                x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
                cropped = frame[y1:y2, x1:x2].copy()
                _, buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 80])
                event_data["data"]["image"] = base64.b64encode(buf).decode("utf-8")
            except Exception as e:
                logger.error(f"Failed to encode NG image for websocket: {e}")
                
        ws_manager.broadcast_to_room_from_thread(main_loop, event_data, str(cam.id))

    def on_frame(annotated_frame):
        _, buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64_frame = base64.b64encode(buf).decode("utf-8")
        frame_data = {"type": "frame", "data": b64_frame}
        ws_manager.broadcast_to_room_from_thread(main_loop, frame_data, str(cam.id))

    pipeline.add_callback("on_result", on_result)
    pipeline.add_callback("on_frame", on_frame)

    pipeline.start()
    request.app.state.pipeline = pipeline
    request.app.state.pipeline_camera_id = cam.id

    logger.info(f"Inference pipeline started for camera {cam.id} with model {model.name}")
    return {"status": "started", "camera_id": cam.id, "model_id": model.id}

@router.post("/stop")
def stop_inference(request: Request):
    # This function is correct and remains unchanged
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline:
        pipeline.stop()
        request.app.state.pipeline = None
        request.app.state.pipeline_camera_id = None
    logger.info("Inference pipeline stopped.")
    return {"status": "stopped"}

@router.get("/stats")
def get_inference_stats(request: Request):
    # This function is correct and remains unchanged
    pipeline = getattr(request.app.state, "pipeline", None)
    is_running = bool(pipeline and pipeline.is_running())
    stats = pipeline.get_stats() if is_running else {}
    return {
        "is_running": is_running,
        "camera_id": getattr(request.app.state, "pipeline_camera_id", None) if is_running else None,
        "total_inspected": int(stats.get("total_inspected", 0)),
        "ok_count": int(stats.get("total_ok", 0)),
        "ng_count": int(stats.get("total_ng", 0)),
        "fps": round(stats.get("fps", 0.0), 1),
    }
