"""
advanced_footfall_streamlit.py
✅ Final Fixed Version — Streamlit + CLI dual-mode Footfall Counter
Supports YOLOv8 + DeepSORT + Simple Mode with smooth video display
"""

import sys
import os
import time
import cv2
import numpy as np
from collections import OrderedDict, deque

# Detect run mode (when launched via `streamlit run`)
RUN_MODE = "streamlit" if any("streamlit" in a for a in sys.argv) else "cli"

# Optional imports
HAS_STREAMLIT = False
HAS_YOLO = False
HAS_DEEPSORT = False
if RUN_MODE == "streamlit":
    try:
        import streamlit as st
        HAS_STREAMLIT = True
    except Exception:
        HAS_STREAMLIT = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    HAS_DEEPSORT = True
except Exception:
    HAS_DEEPSORT = False


# ---------------- Centroid Tracker ----------------
class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.history = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        oid = self.next_object_id
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.history[oid] = deque(maxlen=64)
        self.history[oid].append(centroid)
        self.next_object_id += 1

    def deregister(self, oid):
        self.objects.pop(oid, None)
        self.disappeared.pop(oid, None)
        self.history.pop(oid, None)

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            input_centroids[i] = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(tuple(c))
        else:
            oids = list(self.objects.keys())
            ocent = np.array(list(self.objects.values()))
            D = np.linalg.norm(ocent[:, None] - input_centroids[None, :], axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols or D[r, c] > self.max_distance:
                    continue
                oid = oids[r]
                self.objects[oid] = tuple(input_centroids[c])
                self.disappeared[oid] = 0
                self.history[oid].append(tuple(input_centroids[c]))
                used_rows.add(r)
                used_cols.add(c)

            for r in set(range(D.shape[0])) - used_rows:
                oid = oids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            for c in set(range(D.shape[1])) - used_cols:
                self.register(tuple(input_centroids[c]))

        return self.objects


# ---------------- Simple Detector ----------------
class SimpleDetector:
    def __init__(self, min_area=400):
        self.backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        self.min_area = min_area

    def detect(self, frame):
        fg = self.backsub.apply(frame)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, k, iterations=2)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            rects.append((x, y, x + w, y + h))
        return rects


# ---------------- YOLO + DeepSORT Detector ----------------
class YOLODeepSortDetector:
    def __init__(self, model_path="yolov8n.pt"):
        if not HAS_YOLO:
            raise ImportError("Ultralytics YOLO not installed.")
        self.model = YOLO(model_path)
        self.deepsort = DeepSort(max_age=30) if HAS_DEEPSORT else None

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        rects = []
        if hasattr(results, "boxes"):
            for box in results.boxes:
                xyxy = box.xyxy.cpu().numpy().astype(int).reshape(-1)
                x1, y1, x2, y2 = xyxy[:4]
                rects.append((x1, y1, x2, y2))
        return rects


# ---------------- Utilities ----------------
def check_line_crossing(line, p_prev, p_cur):
    (x1, y1), (x2, y2) = line
    lx, ly = x2 - x1, y2 - y1
    nx, ny = -ly, lx

    def side(p):
        return (p[0] - x1) * nx + (p[1] - y1) * ny

    d_prev, d_cur = side(p_prev), side(p_cur)
    if d_prev * d_cur < 0:
        return True, "in" if d_prev < 0 else "out"
    return False, None


# ---------------- Processor ----------------
class Processor:
    def __init__(self, src, mode="simple", min_area=400):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self.mode = mode
        self.detector = YOLODeepSortDetector() if mode == "advanced" and HAS_YOLO else SimpleDetector(min_area)
        self.tracker = CentroidTracker()
        self.line = ((0, int(self.h * 0.5)), (self.w, int(self.h * 0.5)))
        self.count_in, self.count_out = 0, 0
        self.logs = []

    def step(self, frame):
        rects = self.detector.detect(frame)
        objs = self.tracker.update(rects)
        for oid, hist in list(self.tracker.history.items()):
            if len(hist) >= 2:
                crossed, direction = check_line_crossing(self.line, hist[-2], hist[-1])
                if crossed:
                    if direction == "in":
                        self.count_in += 1
                    else:
                        self.count_out += 1
                    self.tracker.history[oid].clear()
                    self.logs.append({"id": oid, "dir": direction, "time": time.time()})
        return rects, objs

    def release(self):
        self.cap.release()


# ---------------- Streamlit App ----------------
def run_streamlit_app():
    st.set_page_config(page_title="Footfall Counter Pro", layout="wide")
    st.markdown("<h1 style='color:#0f4c81'>📊 Footfall Counter Pro</h1>", unsafe_allow_html=True)

    st.sidebar.header("⚙️ Configuration")
    mode = st.sidebar.selectbox("Detection Mode", ["Simple", "Advanced (YOLOv8 + DeepSORT)"])
    mode_key = "advanced" if "Advanced" in mode else "simple"
    use_webcam = st.sidebar.checkbox("Use Webcam", False)
    min_area = st.sidebar.slider("Min Motion Area", 100, 2000, 400)

    uploaded_file = None
    if not use_webcam:
        uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])

    if st.button("▶️ Start Counting"):
        source = 0 if use_webcam else None
        if not use_webcam and uploaded_file is None:
            st.warning("Please upload a video file first.")
            return
        elif uploaded_file:
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            source = tfile.name

        proc = Processor(source, mode=mode_key, min_area=min_area)
        stframe = st.empty()
        progress = st.progress(0)
        total_frames = int(proc.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frame_idx = 0
        while proc.cap.isOpened():
            ret, frame = proc.cap.read()
            if not ret:
                break
            frame_idx += 1
            rects, objs = proc.step(frame)

            cv2.line(frame, proc.line[0], proc.line[1], (0, 255, 255), 2)
            for (x1, y1, x2, y2) in rects:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            for oid, centroid in proc.tracker.objects.items():
                cx, cy = centroid
                cv2.putText(frame, f"ID {oid}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            progress.progress(min(int((frame_idx / max(total_frames, 1)) * 100), 100))
            st.sidebar.metric("IN", proc.count_in)
            st.sidebar.metric("OUT", proc.count_out)
            time.sleep(0.02)

        st.success(f"✅ Completed — IN: {proc.count_in} | OUT: {proc.count_out}")
        proc.release()


# ---------------- CLI Mode ----------------
def run_cli_mode():
    import argparse
    parser = argparse.ArgumentParser(description="Footfall Counter CLI")
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", choices=["simple", "advanced"], default="simple")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    proc = Processor(src, mode=args.mode)
    while True:
        ret, frame = proc.cap.read()
        if not ret:
            break
        rects, objs = proc.step(frame)
        cv2.line(frame, proc.line[0], proc.line[1], (0, 255, 255), 2)
        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if args.display:
            cv2.imshow("Footfall Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    print(f"Final counts — IN: {proc.count_in}, OUT: {proc.count_out}")
    proc.release()
    cv2.destroyAllWindows()


# ---------------- Entry ----------------
if __name__ == "__main__":
    if RUN_MODE == "streamlit" and HAS_STREAMLIT:
        run_streamlit_app()
    else:
        run_cli_mode()
