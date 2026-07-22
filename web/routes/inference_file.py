"""File-based inference API — run detection + classification on uploaded images/videos."""

import logging
import os
import cv2
import base64
import tempfile
import shutil
import numpy as np
import yaml
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from web.database import get_db, Model

# Import the correct, new classifier and detector from core/
from core.detector import TokenDetector
from core.embedder import TokenEmbedder
from core.classifier import AnomalyClassifier
from core.tracker import ConveyorTracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference-file"])

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Lazy-loaded singletons
_detector = None
_embedder = None
_classifier = None
_debug_config = {}

def _load_debug_config():
    global _debug_config
    if not _debug_config:
        try:
            with open("pipeline_config.yaml", "r") as f:
                _debug_config = yaml.safe_load(f).get("debugging", {})
        except Exception:
            _debug_config = {} # Ensure it's a dict
    return _debug_config

def _get_detector():
    global _detector
    if _detector is None:
        try:
            with open("pipeline_config.yaml", "r") as f:
                cfg = yaml.safe_load(f).get("detection", {})
            _detector = TokenDetector(
                model_path=cfg.get("model_path", "yolo11n_poc_classification_2107.pt"),
                conf_threshold=cfg.get("confidence_threshold", 0.5),
                device=cfg.get("device", "cpu")
            )
            logger.info("Successfully loaded core.detector.TokenDetector.")
        except Exception as e:
            logger.error(f"Failed to load TokenDetector: {e}")
            raise HTTPException(status_code=500, detail="Detector model could not be loaded.")
    return _detector

def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            with open("pipeline_config.yaml", "r") as f:
                cfg = yaml.safe_load(f).get("embedder", {})
            _embedder = TokenEmbedder(device=cfg.get("device", "cpu"))
            logger.info("Successfully loaded core.embedder.TokenEmbedder.")
        except Exception as e:
            logger.error(f"Failed to load TokenEmbedder: {e}")
            raise HTTPException(status_code=500, detail="Embedder could not be loaded.")
    return _embedder

def _get_classifier(model_id: int, db: Session):
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    try:
        with open("pipeline_config.yaml", "r") as f:
            cfg = yaml.safe_load(f).get("classification", {})
        classifier = AnomalyClassifier(threshold=cfg.get("similarity_threshold", 0.75))
        if os.path.exists(model.file_path):
            classifier.load_references(model.file_path)
        else:
            logger.warning(f"Model path does not exist: {model.file_path}")
        return classifier
    except Exception as e:
        logger.error(f"Failed to load AnomalyClassifier: {e}")
        raise HTTPException(status_code=500, detail="Classifier could not be loaded.")

def _frame_to_base64(frame, quality=85):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


@router.post("/image")
async def infer_image(
    files: List[UploadFile] = File(...),
    model_id: int = Form(...),
    db: Session = Depends(get_db)
):
    detector = _get_detector()
    embedder = _get_embedder()
    classifier = _get_classifier(model_id, db)

    all_results = []

    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            all_results.append({
                "filename": file.filename,
                "error": "Cannot decode image",
                "ok_count": 0,
                "ng_count": 0,
                "tokens_count": 0,
                "detections": []
            })
            continue

        annotated = frame.copy()
        
        # 1. Detect obj tokens inside the image using YOLO
        tokens, conveyor_bbox = detector.detect(frame)
        
        detections = []
        ok_count = 0
        ng_count = 0

        # If YOLO found tokens inside the image/conveyor
        if tokens:
            for i, token in enumerate(tokens):
                x1, y1, x2, y2 = token['bbox']
                
                # Crop with padding for better context and accuracy
                cropped = detector.crop_detection(frame, token['bbox'], padding=0.15)
                if cropped.size > 0:
                    emb = embedder.extract(cropped)
                    if emb is not None:
                        res = classifier.classify(emb)
                        result_label = res['result']
                        similarity = res['similarity']
                    else:
                        result_label = "NG"
                        similarity = 0.0
                else:
                    result_label = "NG"
                    similarity = 0.0

                if result_label == "OK":
                    ok_count += 1
                    color = (0, 255, 0)
                else:
                    ng_count += 1
                    color = (0, 0, 255)

                label = f"#{i+1} {result_label} ({similarity*100:.1f}%)"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                detections.append({
                    "id": i + 1,
                    "result": result_label,
                    "similarity": round(similarity, 4),
                    "bbox": [x1, y1, x2, y2]
                })
        else:
            # Fallback if no specific token bounding box was detected by YOLO:
            # Inform user no obj token found, but crop center product area if available
            logger.warning(f"No 'obj' tokens detected in uploaded image {file.filename}.")
            # Assume image center crop
            h, w = frame.shape[:2]
            side = min(h, w)
            cx, cy = w // 2, h // 2
            crop_box = (max(0, cx - side//2), max(0, cy - side//2), min(w, cx + side//2), min(h, cy + side//2))
            cropped = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
            
            if cropped.size > 0:
                emb = embedder.extract(cropped)
                res = classifier.classify(emb) if emb is not None else {'result': 'NG', 'similarity': 0.0}
                result_label = res['result']
                similarity = res['similarity']
            else:
                result_label = "NG"
                similarity = 0.0

            color = (0, 255, 0) if result_label == "OK" else (0, 0, 255)
            if result_label == "OK": ok_count = 1
            else: ng_count = 1

            cv2.putText(annotated, f"{result_label} ({similarity*100:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            detections.append({
                "id": 1,
                "result": result_label,
                "similarity": round(similarity, 4),
                "bbox": list(crop_box)
            })

        cv2.putText(annotated, f"Tokens: {len(detections)} | OK: {ok_count} | NG: {ng_count}", (10, annotated.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        all_results.append({
            "filename": file.filename,
            "ok_count": ok_count,
            "ng_count": ng_count,
            "tokens_count": len(detections),
            "detections": detections,
            "annotated_image": _frame_to_base64(annotated)
        })

    return {"results": all_results}

@router.post("/video")
async def infer_video(
    file: UploadFile = File(...),
    model_id: int = Form(...),
    frame_skip: int = Form(default=5),
    max_frames: int = Form(default=10000),
    db: Session = Depends(get_db)
):
    detector = _get_detector()
    embedder = _get_embedder()
    classifier = _get_classifier(model_id, db)
    debug_cfg = _load_debug_config()

    save_crops = debug_cfg.get("save_cropped_objects", False)
    crop_dir = os.path.join(BASE_DIR, debug_cfg.get("cropped_objects_dir", "data/debug/cropped"))
    
    if save_crops:
        os.makedirs(crop_dir, exist_ok=True)
        # Clean up old debug images
        for old_file in os.listdir(crop_dir):
            os.remove(os.path.join(crop_dir, old_file))
        logger.info(f"Saving cropped objects for debugging to: {crop_dir}")

    # ... (video setup code remains the same)
    tmp_dir = os.path.join(BASE_DIR, "data", "tmp")
    exports_dir = os.path.join(BASE_DIR, "data", "exports")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(exports_dir, exist_ok=True)

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    tmp_path = os.path.join(tmp_dir, f"upload_{ts_str}{suffix}")
    export_filename = f"result_{ts_str}.mp4"
    export_path = os.path.join(exports_dir, export_filename)

    cap = None
    writer = None

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(export_path, fourcc, fps, (width, height))

        tracker = ConveyorTracker(max_disappeared=15, decision_line_y=0.75)
        
        total_ok = 0
        total_ng = 0
        frame_idx = 0
        processed_count = 0
        last_annotated = None
        
        while True:
            ret, frame = cap.read()
            if not ret or processed_count >= max_frames:
                break

            if frame_idx % (frame_skip + 1) != 0:
                if last_annotated is not None: writer.write(last_annotated)
                else: writer.write(frame)
                frame_idx += 1
                continue

            tokens, conveyor_bbox = detector.detect(frame)
            tracker.update(tokens, height, conveyor_bbox)

            for track in tracker.get_tracks_needing_classification():
                cropped = detector.crop_detection(frame, track.bbox, padding=0)
                if cropped.size > 0:
                    emb = embedder.extract(cropped)
                    res = classifier.classify(emb)
                    track.classification_result = res['result']
                    track.classification_similarity = res['similarity']
                    track.classification_done = True
                    
                    if track.classification_result == "OK":
                        total_ok += 1
                    else:
                        total_ng += 1
                    
                    # --- SAVE CROPPED IMAGE FOR DEBUGGING ---
                    if save_crops:
                        try:
                            img_name = f"frame{frame_idx}_track{track.track_id}_{res['result']}_{res['similarity']:.3f}.jpg"
                            cv2.imwrite(os.path.join(crop_dir, img_name), cropped)
                        except Exception as e:
                            logger.warning(f"Could not save debug crop image: {e}")
                    # -----------------------------------------

                else:
                    track.classification_done = True
                    track.classification_result = "NG"
                    total_ng += 1

            # ... (drawing and writing video frame code remains the same)
            annotated = frame.copy()
            for track in tracker.tracks.values():
                x1, y1, x2, y2 = track.bbox
                color = (0, 255, 0) if track.classification_result == "OK" else (0, 0, 255) if track.classification_result == "NG" else (0, 255, 255)
                label = f"ID:{track.track_id}"
                if track.classification_done:
                    label += f" {track.classification_result}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(annotated, f"OK:{total_ok}  NG:{total_ng}  Fr:{frame_idx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            writer.write(annotated)
            last_annotated = annotated
            
            processed_count += 1
            frame_idx += 1

        return {"export_url": f"/storage/exports/{export_filename}", "summary": {"ok_count": total_ok, "ng_count": total_ng}}

    finally:
        if cap: cap.release()
        if writer: writer.release()
        if os.path.exists(tmp_path): os.remove(tmp_path)
