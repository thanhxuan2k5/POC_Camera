import logging
import threading
import time
import cv2
import queue
from collections import defaultdict

from .capture import FrameCapture
from .detector import TokenDetector
from .embedder import TokenEmbedder
from .classifier import AnomalyClassifier
from .tracker import ConveyorTracker

logger = logging.getLogger(__name__)

class InspectionPipeline:
    def __init__(self, config):
        self.config = config
        
        self.capture = FrameCapture(
            url=config.get('rtsp_url'),
            use_gstreamer=config.get('use_gstreamer', False), # Default to False for broader compatibility
            frame_skip=config.get('frame_skip', 0)
        )
        
        self.detector = TokenDetector(
            model_path=config.get('yolo_model'),
            conf_threshold=config.get('conf_threshold'),
            device=config.get('device')
        )
        
        self.embedder = TokenEmbedder(
            device=config.get('device')
        )
        
        self.classifier = AnomalyClassifier(
            threshold=config.get('anomaly_threshold')
        )
        
        if config.get('references_dir'):
            self.classifier.load_references(config['references_dir'])
            
        self.tracker = ConveyorTracker(
            max_disappeared=config.get('max_disappeared'),
            max_distance=config.get('max_distance'),
            decision_line_y=config.get('decision_line_y')
        )
        
        self.callbacks = defaultdict(list)
        self._stop_event = threading.Event()
        self._process_thread = None
        
        self.stats = defaultdict(float)
        self._stats_lock = threading.Lock()
        
    def add_callback(self, event_name, callback_fn):
        self.callbacks[event_name].append(callback_fn)
            
    def start(self):
        self._stop_event.clear()
        with self._stats_lock:
            self.stats['start_time'] = time.time()
            
        self.capture.start()
        
        # Give the capture thread a moment to connect
        time.sleep(1.0) 
        
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True, name="PipelineThread")
        self._process_thread.start()
        
        logger.info("Inspection Pipeline thread started.")
        
    def stop(self):
        self._stop_event.set()
        self.capture.stop()
        
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=5.0)
            
        logger.info("Inspection Pipeline stopped.")
        
    def is_running(self):
        return self._process_thread is not None and self._process_thread.is_alive()
        
    def get_stats(self):
        with self._stats_lock:
            stats = self.stats.copy()
            if stats['start_time']:
                stats['uptime'] = time.time() - stats['start_time']
            return stats
            
    def _draw_overlay(self, frame, tracks, conveyor_bbox=None):
        # Draw vertical QC line
        h, w = frame.shape[:2]
        decision_x = int(self.tracker.get_decision_line_x_px(w))
        
        y_start = 0
        y_end = h
        
        if conveyor_bbox:
            cx1, cy1, cx2, cy2 = conveyor_bbox
            # Vẽ viền cho băng chuyền (màu cam)
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 165, 255), 2)
            cv2.putText(frame, "Conveyor", (cx1, max(cy1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            # Chỉ kẻ vạch QC trong phạm vi chiều cao của băng chuyền
            y_start = cy1
            y_end = cy2

        cv2.line(frame, (decision_x, y_start), (decision_x, y_end), (0, 0, 255), 2)
        cv2.putText(frame, "QC Line", (decision_x + 5, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for track in tracks:
            # Only draw tracks that are stable (hits >= 3) to avoid ghost boxes
            if track.hits < 3 and not track.classification_done:
                continue
                
            x1, y1, x2, y2 = track.bbox
            color = (0, 255, 0) if track.classification_result == "OK" else (0, 0, 255) if track.classification_result == "NG" else (0, 255, 255)
            label = f"ID:{track.track_id}"
            if track.classification_done:
                label += f" {track.classification_result}"
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        with self._stats_lock:
            fps, ok, ng = self.stats["fps"], self.stats["total_ok"], self.stats["total_ng"]
        cv2.putText(frame, f"FPS: {fps:.1f} | OK: {int(ok)} | NG: {int(ng)}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
        
    def _process_loop(self):
        frame_count = 0
        fps_start_time = time.time()
        last_frame_time = time.time()

        while not self._stop_event.is_set():
            try:
                frame, timestamp = self.capture.get_frame(timeout=1.0)
            except queue.Empty:
                if time.time() - last_frame_time > 5.0:
                    logger.error("No frames received from camera for 5 seconds. Stopping pipeline.")
                    self._stop_event.set()
                continue
                
            if frame is None:
                continue
            
            last_frame_time = time.time()
            h, w = frame.shape[:2]
            
            try:
                tokens, conveyor_bbox = self.detector.detect(frame)
                self.tracker.update(tokens, frame_width=w, frame_height=h, conveyor_bbox=conveyor_bbox)
                
                # 1. Area 1: Collect votes continuously
                for track in self.tracker.get_tracks_needing_classification():
                    cropped = self.detector.crop_detection(frame, track.bbox, padding=0.05)
                    if cropped.size > 0:
                        emb = self.embedder.extract(cropped)
                        if emb is not None:
                            cls_res = self.classifier.classify(emb)
                            track.votes.append(cls_res)
                            # Keep queue size limited
                            if len(track.votes) > self.tracker.max_votes:
                                track.votes.pop(0)
                            # Update last crop for UI
                            track.last_crop = cropped
                            
                    track.needs_classification = False
                    
                # 2. Area 2: Make final decision and broadcast
                for track in self.tracker.get_tracks_needing_final_decision():
                    if len(track.votes) > 0:
                        # Thay vì Majority Vote, ta dùng MAX Similarity.
                        # Vì trong Anomaly Detection, nếu có ít nhất 1 frame có độ tự tin cao (vật thể hiển thị rõ nhất)
                        # thì đó là hàng OK. Các frame mờ/nhòe sẽ kéo tụt kết quả nếu dùng Majority.
                        best_vote = max(track.votes, key=lambda v: v['similarity'])
                        final_res = best_vote['result']
                    else:
                        # Fallback if no votes collected (e.g. moved too fast)
                        cropped = self.detector.crop_detection(frame, track.bbox, padding=0.05)
                        final_res = 'NG'
                        if cropped.size > 0:
                            emb = self.embedder.extract(cropped)
                            if emb is not None:
                                cls_res = self.classifier.classify(emb)
                                final_res = cls_res['result']
                            track.last_crop = cropped
                            
                    track.classification_result = final_res
                    track.classification_done = True
                    track.needs_final_decision = False
                    
                    with self._stats_lock:
                        self.stats['total_inspected'] += 1
                        if final_res == 'OK':
                            self.stats['total_ok'] += 1
                        else:
                            self.stats['total_ng'] += 1
                    
                    if self.callbacks['on_result']:
                        # Prepare image to send to UI
                        img_to_send = track.last_crop if track.last_crop is not None else np.zeros((10,10,3), dtype=np.uint8)
                        _, buf = cv2.imencode('.jpg', img_to_send, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        import base64
                        b64_img = base64.b64encode(buf).decode('utf-8')
                        
                        # Use the best vote's similarity for UI
                        if track.votes:
                            avg_sim = best_vote['similarity']
                        else:
                            avg_sim = cls_res['similarity'] if 'cls_res' in locals() else 0.0
                            
                        for cb in self.callbacks['on_result']:
                            cb(track.track_id, final_res, avg_sim, frame, track.bbox, b64_img)

            except Exception as e:
                logger.error(f"Error during detection/classification loop: {e}", exc_info=True)
                continue # Continue to next frame
                
            # FPS Calculation
            frame_count += 1
            if time.time() - fps_start_time > 1.0:
                with self._stats_lock:
                    self.stats['fps'] = frame_count / (time.time() - fps_start_time)
                frame_count = 0
                fps_start_time = time.time()
                
            # Frame Callback
            if self.callbacks['on_frame']:
                annotated_frame = self._draw_overlay(frame.copy(), list(self.tracker.tracks.values()), conveyor_bbox)
                for cb in self.callbacks['on_frame']:
                    cb(annotated_frame)
        
        logger.info("Pipeline processing loop finished.")
