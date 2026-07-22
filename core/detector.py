import logging
import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class TokenDetector:
    def __init__(self, model_path='yolo11n_poc_classification_2107.pt', conf_threshold=0.5, device='cpu'):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self.fixed_conveyor_bbox = None  # Cache conveyor ROI once detected (Jetson Nano optimization)
        
        logger.info(f"Loading YOLO model from {model_path} on {device} with conf_threshold={self.conf_threshold}")
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
            
    def set_fixed_conveyor(self, bbox):
        """Manually lock or update fixed conveyor bbox ROI."""
        self.fixed_conveyor_bbox = bbox
        logger.info(f"Fixed conveyor ROI set to {bbox}")

    def reset_fixed_conveyor(self):
        """Reset fixed conveyor ROI so it can be re-detected."""
        self.fixed_conveyor_bbox = None

    def detect(self, frame):
        """
        Detects 'obj' and 'bangchuyen'.
        - Locks conveyor ROI on first detection for Jetson Nano speed optimization.
        - Filters 'obj' tokens within the conveyor polygon.
        """
        if self.fixed_conveyor_bbox:
            # Optimize for Jetson: only detect class 0 (obj) since conveyor is known
            results = self.model.predict(frame, conf=self.conf_threshold, iou=0.45, classes=[0], verbose=False, device=self.device)
        else:
            results = self.model.predict(frame, conf=self.conf_threshold, iou=0.45, verbose=False, device=self.device)
        
        all_objs = []
        detected_conveyor_bbox = None
        highest_conv_conf = 0.0
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                
                if cls_id == 0:  # obj (wood token)
                    w = x2 - x1
                    h = y2 - y1
                    aspect_ratio = w / h if h > 0 else 0
                    # Filter out ghost detections (too long/tall or too small)
                    if 0.6 <= aspect_ratio <= 1.6 and w > 20 and h > 20:
                        all_objs.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': conf,
                            'class_id': 0,
                            'class_name': 'obj'
                        })
                elif cls_id == 1:  # bangchuyen (conveyor)
                    if conf > highest_conv_conf:
                        highest_conv_conf = conf
                        detected_conveyor_bbox = (int(x1), int(y1), int(x2), int(y2))
        
        # If we detect conveyor and don't have fixed_conveyor_bbox yet, lock it
        if detected_conveyor_bbox is not None and self.fixed_conveyor_bbox is None:
            self.fixed_conveyor_bbox = detected_conveyor_bbox
            logger.info(f"Conveyor ROI locked on frame detection: {self.fixed_conveyor_bbox}")

        effective_conveyor = detected_conveyor_bbox or self.fixed_conveyor_bbox

        # Sort all found objects by confidence, descending
        all_objs.sort(key=lambda x: x['confidence'], reverse=True)
        
        # --- Filtering Logic ---
        if effective_conveyor is not None:
            cx1, cy1, cx2, cy2 = effective_conveyor
            filtered_tokens = []
            for t in all_objs:
                tx1, ty1, tx2, ty2 = t['bbox']
                token_center_x = (tx1 + tx2) / 2
                token_center_y = (ty1 + ty2) / 2
                if (cx1 - 20) <= token_center_x <= (cx2 + 20) and \
                   (cy1 - 20) <= token_center_y <= (cy2 + 20):
                    filtered_tokens.append(t)
            
            final_tokens = filtered_tokens
        else:
            final_tokens = all_objs
        
        return final_tokens, effective_conveyor
        
    def crop_detection(self, frame, detection, padding=0):
        """
        Crops a detection from the frame with a square aspect ratio.
        Accepts bbox tuple (x1, y1, x2, y2), dict with 'bbox' key, or Track object with .bbox attribute.
        """
        if hasattr(detection, 'bbox'):
            bbox = detection.bbox
        elif isinstance(detection, dict) and 'bbox' in detection:
            bbox = detection['bbox']
        elif isinstance(detection, (tuple, list)):
            bbox = detection
        else:
            logger.error(f"Invalid detection object passed to crop_detection: {type(detection)}")
            return np.empty((0, 0, 3), dtype=np.uint8)

        x1, y1, x2, y2 = bbox
        h_img, w_img = frame.shape[:2]
        
        # Calculate center and side length
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        cx = x1 + w / 2
        cy = y1 + h / 2
        
        # Make crop square to match training crop style, add padding
        side = max(w, h) * (1 + padding)
        
        # Calculate new square coordinates, ensuring they are within image bounds
        nx1 = int(max(0, cx - side / 2))
        ny1 = int(max(0, cy - side / 2))
        nx2 = int(min(w_img, cx + side / 2))
        ny2 = int(min(h_img, cy + side / 2))
        
        if nx2 <= nx1 or ny2 <= ny1:
            return np.empty((0, 0, 3), dtype=np.uint8)

        cropped = frame[ny1:ny2, nx1:nx2].copy()
        return cropped
