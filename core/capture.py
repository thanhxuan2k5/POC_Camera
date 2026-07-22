import cv2
import time
import threading
import queue
import logging

logger = logging.getLogger(__name__)

class FrameCapture:
    def __init__(self, url, use_gstreamer=True, frame_skip=0, max_queue_size=2):
        self.url = url
        self.use_gstreamer = use_gstreamer
        self.frame_skip = frame_skip
        self.max_queue_size = max_queue_size
        
        self._frame_queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread = None
        self._cap = None
        
    def _get_gstreamer_pipeline(self):
        return (
            f"rtspsrc location={self.url} latency=200 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! "
            "video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1"
        )
        
    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CaptureThread")
        self._thread.start()
        logger.info("FrameCapture started.")
        
    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info("FrameCapture stopped.")
        
    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()
        
    def get_frame(self, timeout=None):
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None
            
    def _capture_loop(self):
        backoff = 1
        max_backoff = 30
        
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                if self.use_gstreamer:
                    pipeline = self._get_gstreamer_pipeline()
                    logger.info(f"Opening stream with GStreamer: {pipeline}")
                    self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                else:
                    logger.info(f"Opening stream with FFMPEG: {self.url}")
                    self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    
                if not self._cap.isOpened():
                    logger.warning(f"Failed to open stream. Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
                    continue
                else:
                    logger.info("Stream opened successfully.")
                    backoff = 1
            
            frame_count = 0
            while not self._stop_event.is_set():
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning("Failed to read frame. Reconnecting...")
                    self._cap.release()
                    self._cap = None
                    break
                
                frame_count += 1
                if self.frame_skip > 0 and frame_count % (self.frame_skip + 1) != 0:
                    continue
                
                timestamp = time.time()
                
                # Clear old frames if queue is full
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                try:
                    self._frame_queue.put((frame, timestamp), timeout=0.1)
                except queue.Full:
                    pass
