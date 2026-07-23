"""
ConveyorTracker: tracks 'obj' tokens on the conveyor belt.

Key design decisions:
  - YOLO runs ONCE per frame on the full image → returns both class-0 (obj) and class-1 (bangchuyen).
  - The conveyor bbox (smoothed with EMA) defines a polygon ROI.
  - Only tokens whose centroid is INSIDE that polygon are tracked (cv2.pointPolygonTest).
  - When a token crosses the decision line (default 75% of conveyor height), it is flagged
    for classification. The caller then crops that token from the ORIGINAL frame and runs
    the embedder on it — no second YOLO pass needed.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class Track:
        "track_id", "bbox", "centroid_history",
        "classification_result", "classification_done",
        "needs_classification", "disappeared_frames", "hits",
        "needs_final_decision", "votes", "last_crop"

    def __init__(self, track_id, bbox, centroid):
        self.track_id = track_id
        self.bbox = bbox                      # (x1,y1,x2,y2) in full-frame coords
        self.centroid_history = [centroid]
        self.classification_result = None     # 'OK' | 'NG'
        self.classification_done = False
        self.needs_classification = False
        self.needs_final_decision = False
        self.votes = []
        self.last_crop = None
        self.disappeared_frames = 0
        self.hits = 1

    @property
    def current_centroid(self):
        return self.centroid_history[-1]


class ConveyorTracker:
    def __init__(self, max_disappeared=10, max_distance=100.0, decision_line_y=0.8, ema_alpha=0.25, max_votes=15):
        """
        decision_line_y : fraction of conveyor width at which a token is finalized (0.0-1.0).
        ema_alpha       : smoothing factor for conveyor bbox EMA.
        max_votes       : max queue size for classification votes.
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.decision_line_y = decision_line_y
        self.ema_alpha = ema_alpha
        self.max_votes = max_votes

        self.next_track_id = 0
        self.tracks: dict[int, Track] = {}
        self.finalized_tracks: list[Track] = []

        # Smoothed conveyor bounding box (x1, y1, x2, y2) — updated by EMA each frame.
        self.conveyor_bbox = None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _conveyor_poly(self):
        """Return the conveyor bbox as an Nx1x2 int32 array for cv2 functions, or None."""
        if self.conveyor_bbox is None:
            return None
        x1, y1, x2, y2 = self.conveyor_bbox
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

    def _inside_conveyor(self, cx, cy):
        """True if (cx, cy) is strictly inside the conveyor polygon."""
        poly = self._conveyor_poly()
        if poly is None:
            return False  # Strictly require conveyor to be detected
        return cv2.pointPolygonTest(poly, (float(cx), float(cy)), measureDist=False) >= 0

    def _decision_line_x_px(self, frame_width):
        """Absolute pixel X of the decision line."""
        if self.conveyor_bbox is not None:
            cx1, _, cx2, _ = self.conveyor_bbox
            return cx1 + (cx2 - cx1) * self.decision_line_y
        # Fallback if no conveyor detected
        return frame_width * self.decision_line_y

    # ------------------------------------------------------------------
    # EMA update for conveyor bbox
    # ------------------------------------------------------------------

    def _update_conveyor(self, new_bbox):
        if new_bbox is None:
            return
        if self.conveyor_bbox is None:
            self.conveyor_bbox = tuple(int(v) for v in new_bbox)
        else:
            a = self.ema_alpha
            self.conveyor_bbox = tuple(
                int(c * (1 - a) + n * a)
                for c, n in zip(self.conveyor_bbox, new_bbox)
            )

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, tokens, frame_width, frame_height, conveyor_bbox=None):
        """
        tokens         : list[dict] with keys 'bbox', 'confidence', 'class_id', 'class_name'
                         (class 0 = obj, already pre-filtered by detector)
        frame_width    : full-frame width (px)
        frame_height   : full-frame height (px)
        conveyor_bbox  : (x1,y1,x2,y2) from YOLO class-1 detection, or None
        """
        self.finalized_tracks.clear()

        # 1. Update smoothed conveyor ROI
        self._update_conveyor(conveyor_bbox)

        # 2. Filter tokens: only those whose centroid is INSIDE the conveyor polygon
        filtered = []
        for det in tokens:
            cx, cy = self._centroid(det["bbox"])
            if self._inside_conveyor(cx, cy):
                filtered.append(det)

        # 3. Age out tracks that have no matching detection
        if not filtered:
            for tid in list(self.tracks):
                self.tracks[tid].disappeared_frames += 1
                if self.tracks[tid].disappeared_frames > self.max_disappeared:
                    self.finalized_tracks.append(self.tracks.pop(tid))
            return

        det_centroids = [self._centroid(d["bbox"]) for d in filtered]
        det_bboxes    = [d["bbox"] for d in filtered]

        # 4. If no existing tracks, register and mark for classification immediately
        if not self.tracks:
            for bbox, centroid in zip(det_bboxes, det_centroids):
                self._register(bbox, centroid)
            return

        # 5. Greedy nearest-neighbour matching
        track_ids      = list(self.tracks)
        track_cents    = [self.tracks[tid].current_centroid for tid in track_ids]

        D = np.zeros((len(track_cents), len(det_centroids)), dtype=np.float32)
        for i, tc in enumerate(track_cents):
            for j, dc in enumerate(det_centroids):
                D[i, j] = np.hypot(tc[0] - dc[0], tc[1] - dc[1])

        used_rows, used_cols = set(), set()

        for row, col in zip(*np.unravel_index(np.argsort(D, axis=None), D.shape)):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                break

            tid   = track_ids[row]
            track = self.tracks[tid]

            track.bbox = det_bboxes[col]
            track.centroid_history.append(det_centroids[col])
            if len(track.centroid_history) > 30:
                track.centroid_history = track.centroid_history[-30:]
            track.disappeared_frames = 0
            track.hits += 1

            # Continuous voting in Area 1, finalize when crossing to Area 2
            decision_x = self._decision_line_x_px(frame_width)
            if not track.classification_done:
                if track.current_centroid[0] < decision_x:
                    track.needs_classification = True
                else:
                    track.needs_classification = False
                    if not track.needs_final_decision:
                        track.needs_final_decision = True

            used_rows.add(row)
            used_cols.add(col)

        # Age unmatched tracks
        for i, tid in enumerate(track_ids):
            if i not in used_rows:
                self.tracks[tid].disappeared_frames += 1
                if self.tracks[tid].disappeared_frames > self.max_disappeared:
                    self.finalized_tracks.append(self.tracks.pop(tid))

        # Register new detections
        for j in range(len(det_centroids)):
            if j not in used_cols:
                self._register(det_bboxes[j], det_centroids[j])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _register(self, bbox, centroid):
        track = Track(self.next_track_id, bbox, centroid)
        track.needs_classification = False
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1

    def get_tracks_needing_classification(self):
        return [t for t in self.tracks.values() if t.needs_classification]
        
    def get_tracks_needing_final_decision(self):
        return [t for t in self.tracks.values() if t.needs_final_decision]

    def get_finalized_tracks(self):
        return self.finalized_tracks

    def get_decision_line_x_px(self, frame_width):
        return self._decision_line_x_px(frame_width)

