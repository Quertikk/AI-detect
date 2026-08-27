"""
Video analysis: sample frames, run the face classifier on each detected face,
and aggregate into a video-level verdict.

Face detection uses OpenCV's bundled Haar cascade rather than a heavier
dependency (e.g. mediapipe) - it is less accurate on off-angle faces but
needs no extra model download and works reliably across platforms.

`temporal_jitter_score` is an experimental, unvalidated heuristic: it
measures how much the detected face's position/size fluctuates between
sampled frames, normalized by face size. It is reported for context, not
used to compute the verdict - real videos with camera shake can also score
high here. See README limitations.
"""
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from model import DeepfakeClassifier

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class VideoAnalyzer:
    def __init__(self, classifier: DeepfakeClassifier, frame_interval_sec: float = 1.0, max_frames: int = 30):
        self.classifier = classifier
        self.frame_interval_sec = frame_interval_sec
        self.max_frames = max_frames
        self.face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    def analyze(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file", "frame_results": []}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_step = max(1, int(fps * self.frame_interval_sec))

        frame_results = []
        face_boxes = []
        frame_idx = 0

        while len(frame_results) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                face_box = self._largest_face(frame)
                if face_box is not None:
                    x, y, w, h = face_box
                    face_boxes.append(face_box)
                    crop_rgb = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
                    pred = self.classifier.predict(Image.fromarray(crop_rgb))
                    frame_results.append({
                        "timestamp_sec": round(frame_idx / fps, 2),
                        "prob_fake": pred["prob_fake"],
                        "label": pred["label"],
                    })
            frame_idx += 1

        cap.release()

        if not frame_results:
            return {"error": "No face detected in sampled frames", "frame_results": []}

        fake_scores = [r["prob_fake"] for r in frame_results]
        avg_prob_fake = float(np.mean(fake_scores))

        return {
            "verdict": "fake" if avg_prob_fake >= 0.5 else "real",
            "avg_prob_fake": avg_prob_fake,
            "suspicious_frame_ratio": sum(1 for s in fake_scores if s >= 0.5) / len(fake_scores),
            "temporal_jitter_score": self._bbox_jitter(face_boxes),
            "frame_results": frame_results,
        }

    def _largest_face(self, frame) -> Optional[tuple]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        return tuple(max(faces, key=lambda f: f[2] * f[3]))

    @staticmethod
    def _bbox_jitter(boxes: list) -> Optional[float]:
        if len(boxes) < 2:
            return None
        boxes_arr = np.array(boxes, dtype=np.float32)
        centers = boxes_arr[:, :2] + boxes_arr[:, 2:] / 2
        sizes = np.maximum(boxes_arr[:, 2], boxes_arr[:, 3])
        deltas = np.linalg.norm(np.diff(centers, axis=0), axis=1) / np.maximum(sizes[:-1], 1e-6)
        return float(np.std(deltas))
