#!/usr/bin/env python3
"""
Phaenomena - LIVE DASHBOARD (Multi-Sensor Overview)

Extends emotion_unified_phaenomena.py (kept untouched — used only as a design
and architecture reference) with a single dashboard that shows every live
data source at once:

  - Plant bioelectric voltage  (ESP32 / AD8232, WebSocket)
  - CO2 (ppm)                  (ESP32 / SCD4x, WebSocket)
  - Polar H10 heart rate       (Bluetooth LE)
  - Camera + HSEmotion         (facial emotion recognition)
  - Plant emotion prediction   (pretrained ML model — MOCK for now, see
                                PlantEmotionModel below)

Unlike the exhibition script, there is no on-site training UI and no
continuous model retraining here: the plant-emotion model is loaded once
(or, until a real one exists, fitted once on synthetic data as a stand-in)
and used purely for live inference.

Live data handling follows notebooks/02_init_recording.ipynb (Polar BLE
parsing, ESP32 WebSocket protocol) and arduino/plant_sensor_script.ino
(message format: {"type":"data","voltages":[...],"co2":...}).

Port: 5004
"""

import asyncio
import json
import pickle
import struct
import threading
import time
import urllib.request  # must be imported before hsemotion_onnx (it calls
                        # urllib.request.urlretrieve without importing the
                        # submodule itself)
from collections import deque
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

print("\nChecking optional hardware / ML libraries...")

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    print("  opencv-python not installed -> camera stream disabled")

try:
    import websockets
    WEBSOCKETS_OK = True
except ImportError:
    WEBSOCKETS_OK = False
    print("  websockets not installed -> plant sensor / CO2 stream disabled")

try:
    from bleak import BleakClient, BleakScanner
    BLEAK_OK = True
except ImportError:
    BLEAK_OK = False
    print("  bleak not installed -> Polar heart-rate disabled")

# ===== Face + emotion detection (graceful cascade, mirrors the original
#       script's try/except chain but also works without torch/facenet-pytorch
#       by falling back to OpenCV's built-in Haar cascade) =====
USE_HSEMOTION = False
HSEMOTION_MODEL = None
MTCNN_DETECTOR = None
HAAR_CASCADE = None
FACE_DETECTOR_MODE = None

if CV2_OK:
    try:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer as HSEmotionONNX
        HSEMOTION_MODEL = HSEmotionONNX(model_name='enet_b0_8_best_afew')
        _test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _ = HSEMOTION_MODEL.predict_emotions(_test_img, logits=False)
        USE_HSEMOTION = True
        print("  HSEmotion-ONNX ready")
    except Exception as e:
        print(f"  HSEmotion-ONNX unavailable ({e})")

    if USE_HSEMOTION:
        try:
            from facenet_pytorch import MTCNN
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            MTCNN_DETECTOR = MTCNN(keep_all=False, device=device, min_face_size=60)
            FACE_DETECTOR_MODE = 'mtcnn'
            print("  Face detector: MTCNN (facenet-pytorch)")
        except Exception:
            HAAR_CASCADE = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            FACE_DETECTOR_MODE = 'haar'
            print("  Face detector: OpenCV Haar cascade (facenet-pytorch/torch not installed)")

if not USE_HSEMOTION:
    print("  WARNING: no facial emotion recognition available")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from flask import Flask, Response, render_template_string
from flask_socketio import SocketIO

import logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

ESP32_IP = "10.138.29.238"     # IP shown on the sensor's OLED display
ESP32_PORT = 81                 # see arduino/plant_sensor_script.ino

CAMERA_INDEX_CANDIDATES = [1, 0]   # prefer external/continuity cam, then built-in
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# The ESP32 firmware decimates 380Hz -> "100Hz" via integer division
# (380 // 100 == 3), so the true output rate is 380/3 Hz, not 100Hz.
# Matches notebooks/03_ground_truth_classifier*.ipynb (PLANT_TRUE_FS).
PLANT_TRUE_FS = 380.0 / 3.0
ADC_VREF = 3.3

PLANT_BUFFER_SECONDS = 20          # kept in memory for charting + features
PLANT_WINDOW_SECONDS = 15          # window used for each emotion prediction
PLANT_BUFFER_MAXLEN = int(PLANT_TRUE_FS * PLANT_BUFFER_SECONDS)
PLANT_WINDOW_SAMPLES = int(PLANT_TRUE_FS * PLANT_WINDOW_SECONDS)
MIN_PLANT_SAMPLES_FOR_PREDICTION = int(PLANT_TRUE_FS * 5)

PLANT_PREDICTION_INTERVAL_S = 2.0
CO2_HISTORY_MAXLEN = 180
HR_HISTORY_MAXLEN = 60
CHART_VOLTAGE_POINTS = 400

UPDATE_RATE_MS = 150

# Drop a real model here (dict with 'model' + 'scaler', trained via
# notebooks/05_ml_pipeline*.ipynb on FEATURE_COLUMNS below) to replace the mock.
MODEL_PATH = Path(__file__).parent / "models" / "plant_emotion_model.pkl"

HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


# ═══════════════════════════════════════════════════════════════
# PLANT FEATURE EXTRACTION
# (ported from notebooks/03_ground_truth_classifier_v03.ipynb so that a
#  future real model — trained on that notebook's output — can be dropped
#  in without touching the live pipeline)
# ═══════════════════════════════════════════════════════════════

_trapz = getattr(np, "trapezoid", None) or np.trapz

FEATURE_COLUMNS = [
    "plant_mean", "plant_median", "plant_std", "plant_iqr",
    "plant_min", "plant_max", "plant_range", "plant_rms", "plant_zcr",
    "plant_sat_pos_frac", "plant_sat_neg_frac", "plant_const_flag",
    "bp_0_1", "bp_1_5", "bp_5_15", "bp_15_30", "bp_30_50", "total_power",
    "plant_dead_flag", "plant_suspect_flag",
]
LABELS = ["negativ", "neutral", "positiv"]


def extract_plant_features(volts, fs=PLANT_TRUE_FS):
    """Phaenomena-style features for one window of plant voltage (in V).
    Robust to empty / constant / very short windows."""
    out = {}
    v = np.asarray(volts, dtype=float)
    n = v.size
    out["plant_n_samples"] = int(n)

    if n == 0:
        for k in ["plant_mean", "plant_median", "plant_std", "plant_iqr", "plant_min",
                  "plant_max", "plant_range", "plant_rms", "plant_zcr",
                  "plant_sat_pos_frac", "plant_sat_neg_frac",
                  "bp_0_1", "bp_1_5", "bp_5_15", "bp_15_30", "bp_30_50", "total_power"]:
            out[k] = np.nan
        out["plant_const_flag"] = False
        out["plant_quality"] = "dead_flat"
        out["plant_dead_flag"] = True
        out["plant_suspect_flag"] = False
        return out

    mean = float(np.mean(v)); median = float(np.median(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    q75, q25 = np.percentile(v, [75, 25]); iqr = float(q75 - q25)
    vmin = float(np.min(v)); vmax = float(np.max(v)); rng_ = float(vmax - vmin)
    rms = float(np.sqrt(np.mean(v ** 2)))
    vd = v - mean
    zcr = float(np.mean(np.abs(np.diff(np.sign(vd))) > 0)) if n > 1 else 0.0
    out.update(plant_mean=round(mean, 6), plant_median=round(median, 6),
               plant_std=round(std, 6), plant_iqr=round(iqr, 6),
               plant_min=round(vmin, 6), plant_max=round(vmax, 6),
               plant_range=round(rng_, 6), plant_rms=round(rms, 6),
               plant_zcr=round(zcr, 6))

    sat_pos = float(np.mean(v >= ADC_VREF - 1e-3)); sat_neg = float(np.mean(v <= 1e-3))
    out.update(plant_sat_pos_frac=round(sat_pos, 4),
               plant_sat_neg_frac=round(sat_neg, 4),
               plant_const_flag=bool(np.all(v == v[0])))

    if n >= 16 and std > 0:
        nper = min(256, n)
        f, pxx = sp_signal.welch(vd, fs=fs, window="hann", nperseg=nper, scaling="density")

        def band(lo, hi):
            m = (f >= lo) & (f < hi)
            return float(_trapz(pxx[m], f[m])) if m.any() else 0.0

        bp_0_1 = band(0, 1); bp_1_5 = band(1, 5); bp_5_15 = band(5, 15)
        bp_15_30 = band(15, 30); bp_30_50 = band(30, 50)
        total = bp_0_1 + bp_1_5 + bp_5_15 + bp_15_30 + bp_30_50
    else:
        bp_0_1 = bp_1_5 = bp_5_15 = bp_15_30 = bp_30_50 = total = np.nan
    out.update(bp_0_1=bp_0_1, bp_1_5=bp_1_5, bp_5_15=bp_5_15,
               bp_15_30=bp_15_30, bp_30_50=bp_30_50, total_power=total)

    lsb = ADC_VREF / 4096.0; std_counts = std / lsb
    if sat_pos > 0.95 or sat_neg > 0.95:
        q = "dead_saturated"
    elif std_counts < 50:
        q = "dead_flat"
    elif (sat_pos + sat_neg) > 0.20:
        q = "suspect_clipped"
    elif std_counts < 200:
        q = "suspect_lowamp"
    else:
        q = "ok"
    out["plant_quality"] = q
    out["plant_dead_flag"] = q.startswith("dead")
    out["plant_suspect_flag"] = q.startswith("suspect")
    return out


def _safe_feature_value(features, key):
    val = features.get(key)
    if val is None:
        return 0.0
    val = float(val)
    return 0.0 if val != val else val  # NaN check


# ═══════════════════════════════════════════════════════════════
# PLANT EMOTION MODEL — inference only, never retrained live
# ═══════════════════════════════════════════════════════════════

class PlantEmotionModel:
    """Predicts a plant-derived emotion label from FEATURE_COLUMNS.

    MOCK MODE: no trained model exists yet, so this fits a small RandomForest
    once, at start-up, on synthetic data shaped like extract_plant_features()
    output. It is never touched again while the dashboard runs -- purely a
    stand-in so the dashboard is fully wired up.

    To use the real model: pickle {'model': <fitted classifier>,
    'scaler': <fitted StandardScaler>} trained on FEATURE_COLUMNS via
    notebooks/05_ml_pipeline*.ipynb and save it to MODEL_PATH. It will be
    loaded automatically on the next start and the mock branch is skipped.
    """

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = None
        self.is_mock = True
        if not self._load_real_model():
            self._fit_mock_model()

    def _load_real_model(self):
        if not self.model_path.exists():
            return False
        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.is_mock = False
            print(f"Loaded pretrained plant-emotion model from {self.model_path}")
            return True
        except Exception as e:
            print(f"Could not load pretrained model ({e}); using mock model instead")
            return False

    def _fit_mock_model(self):
        rng = np.random.default_rng(42)
        n = 210
        labels = np.tile(np.array(LABELS), n // len(LABELS) + 1)[:n]
        rng.shuffle(labels)

        X = rng.normal(0, 1, size=(n, len(FEATURE_COLUMNS)))
        # Nudge a couple of columns per label so the mock prediction visibly
        # reacts to the live plant signal instead of being pure noise. This
        # is still not a learned relationship -- just a believable demo.
        label_offset = {"negativ": -1.0, "neutral": 0.0, "positiv": 1.0}
        mean_idx = FEATURE_COLUMNS.index("plant_mean")
        std_idx = FEATURE_COLUMNS.index("plant_std")
        for i, lbl in enumerate(labels):
            X[i, mean_idx] += label_offset[lbl] * 1.5
            X[i, std_idx] += abs(label_offset[lbl]) * 0.8

        self.scaler.fit(X)
        self.model = RandomForestClassifier(n_estimators=60, max_depth=5, random_state=42)
        self.model.fit(self.scaler.transform(X), labels)
        print(f"No pretrained plant-emotion model found at {self.model_path}")
        print("-> using a MOCK model (fit once on synthetic data). Replace it later "
              "with the real one trained in notebooks/05_ml_pipeline*.ipynb.")

    def predict(self, features):
        if self.model is None:
            return None, 0.0, {}
        try:
            x = np.array([[_safe_feature_value(features, c) for c in FEATURE_COLUMNS]])
            x_scaled = self.scaler.transform(x)
            probs = self.model.predict_proba(x_scaled)[0]
            classes = list(self.model.classes_)
            idx = int(np.argmax(probs))
            prob_map = {cls: float(p) for cls, p in zip(classes, probs)}
            return classes[idx], float(probs[idx]), prob_map
        except Exception as e:
            print(f"Plant emotion prediction error: {e}")
            return None, 0.0, {}


# ═══════════════════════════════════════════════════════════════
# CAMERA + HSEMOTION
# ═══════════════════════════════════════════════════════════════

class CameraHandler:
    def __init__(self):
        self.cap = None
        self.available = False
        self.current_frame = None
        self.face_detected = False
        self.current_emotion = None
        self.current_confidence = 0.0
        self.lock = threading.Lock()

    def start(self):
        if not CV2_OK:
            return False
        for index in CAMERA_INDEX_CANDIDATES:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    self.cap = cap
                    self.available = True
                    print(f"Camera started (index {index})")
                    return True
            cap.release()
        return False

    def get_frame(self):
        with self.lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpeg.tobytes() if ret else None

    def process_frame(self):
        if not self.available or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return
        frame = cv2.flip(frame, 1)

        emotion, confidence, box = (None, 0.0, None)
        if USE_HSEMOTION:
            emotion, confidence, box = self._detect_emotion(frame)

        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (168, 255, 28), 2)
            if emotion:
                label = f"{emotion} {confidence * 100:.0f}%"
                cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (168, 255, 28), 2)

        with self.lock:
            self.current_frame = frame
            self.face_detected = box is not None
            if box is not None:
                self.current_emotion = emotion
                self.current_confidence = confidence

    def _detect_emotion(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box = self._find_face(frame, rgb)
            if box is None:
                return None, 0.0, None
            x1, y1, x2, y2 = box
            face_img = rgb[y1:y2, x1:x2]
            if face_img.size == 0:
                return None, 0.0, None
            emotion, scores = HSEMOTION_MODEL.predict_emotions(face_img, logits=False)
            confidence = float(np.max(scores)) if scores is not None and len(scores) else 0.5
            return emotion, confidence, box
        except Exception:
            return None, 0.0, None

    def _find_face(self, frame, rgb):
        h, w = frame.shape[:2]
        if FACE_DETECTOR_MODE == 'mtcnn':
            boxes, _ = MTCNN_DETECTOR.detect(rgb)
            if boxes is None or len(boxes) == 0:
                return None
            x1, y1, x2, y2 = [int(v) for v in boxes[0]]
        elif FACE_DETECTOR_MODE == 'haar':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = HAAR_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
            if len(faces) == 0:
                return None
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh
        else:
            return None
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2


# ═══════════════════════════════════════════════════════════════
# APPLICATION STATE
# ═══════════════════════════════════════════════════════════════

class AppState:
    def __init__(self):
        self.plant_connected = False
        self.voltage_buffer = deque(maxlen=PLANT_BUFFER_MAXLEN)
        self.plant_quality = None

        self.latest_co2 = None
        self.co2_history = deque(maxlen=CO2_HISTORY_MAXLEN)

        self.polar_connected = False
        self.latest_bpm = None
        self.hr_history = deque(maxlen=HR_HISTORY_MAXLEN)

        self.plant_emotion = None
        self.plant_confidence = 0.0
        self.plant_probs = {}


state = AppState()
camera = CameraHandler()
plant_model = PlantEmotionModel()


# ═══════════════════════════════════════════════════════════════
# ESP32 — plant voltage + CO2 (WebSocket, see plant_sensor_script.ino)
# ═══════════════════════════════════════════════════════════════

async def esp32_listener():
    if not WEBSOCKETS_OK:
        return
    uri = f"ws://{ESP32_IP}:{ESP32_PORT}"
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                state.plant_connected = True
                print(f"Connected to ESP32 at {uri}")
                async for message in ws:
                    try:
                        data = json.loads(message)
                    except Exception:
                        continue
                    if data.get("type") != "data":
                        continue
                    for mv in data.get("voltages", []):
                        state.voltage_buffer.append(mv / 1000.0)
                    co2 = data.get("co2")
                    if co2 is not None:
                        state.latest_co2 = float(co2)
                        state.co2_history.append(float(co2))
        except Exception:
            pass
        state.plant_connected = False
        await asyncio.sleep(2)


def run_esp32_listener():
    asyncio.new_event_loop().run_until_complete(esp32_listener())


# ═══════════════════════════════════════════════════════════════
# POLAR H10 — heart rate via BLE (see notebooks/02_init_recording.ipynb)
# ═══════════════════════════════════════════════════════════════

def polar_notification_handler(_, data):
    """Decode Bluetooth SIG Heart Rate Measurement notifications.
    R-R intervals arrive as little-endian uint16 in units of 1/1024s."""
    flags = data[0]
    hr_16_bit = flags & 0x01
    rr_present = flags & 0x10
    offset = 3 if hr_16_bit else 2
    if flags & 0x08:
        offset += 2  # skip Energy Expended field
    if not rr_present:
        return
    while offset + 1 < len(data):
        rr_raw = struct.unpack_from("<H", data, offset)[0]
        rr_ms = rr_raw / 1024 * 1000
        if rr_ms > 0:
            bpm = 60000.0 / rr_ms
            state.latest_bpm = round(bpm, 1)
            state.hr_history.append(state.latest_bpm)
        offset += 2


async def polar_listener():
    if not BLEAK_OK:
        return
    while True:
        try:
            print("Scanning for Polar H10 ...")
            devices = await BleakScanner.discover(timeout=5.0)
            polar = next((d for d in devices if d.name and "Polar" in d.name), None)
            if polar is None:
                await asyncio.sleep(3)
                continue
            async with BleakClient(polar.address) as client:
                await client.start_notify(HR_CHAR_UUID, polar_notification_handler)
                state.polar_connected = True
                print(f"Polar connected: {polar.name}")
                while client.is_connected:
                    await asyncio.sleep(1)
        except Exception as e:
            print(f"Polar disconnected ({e}); retrying ...")
        state.polar_connected = False
        await asyncio.sleep(3)


def run_polar_listener():
    if not BLEAK_OK:
        print("bleak not installed -> Polar heart-rate disabled")
        return
    asyncio.new_event_loop().run_until_complete(polar_listener())


# ═══════════════════════════════════════════════════════════════
# BACKGROUND LOOPS
# ═══════════════════════════════════════════════════════════════

def camera_loop():
    while True:
        camera.process_frame()
        time.sleep(0.03)


def plant_prediction_loop():
    while True:
        samples = list(state.voltage_buffer)
        if len(samples) >= MIN_PLANT_SAMPLES_FOR_PREDICTION:
            window = samples[-PLANT_WINDOW_SAMPLES:]
            features = extract_plant_features(window)
            state.plant_quality = features.get("plant_quality")
            label, confidence, probs = plant_model.predict(features)
            if label is not None:
                state.plant_emotion = label
                state.plant_confidence = confidence
                state.plant_probs = probs
        time.sleep(PLANT_PREDICTION_INTERVAL_S)


def broadcast_updates():
    while True:
        try:
            with camera.lock:
                face_detected = camera.face_detected
                face_emotion = camera.current_emotion
                face_confidence = camera.current_confidence

            socketio.emit('update', {
                'plant_connected': state.plant_connected,
                'camera_available': camera.available,
                'polar_connected': state.polar_connected,

                'voltages': [round(v * 1000, 2) for v in list(state.voltage_buffer)[-CHART_VOLTAGE_POINTS:]],
                'plant_quality': state.plant_quality,

                'co2': state.latest_co2,
                'co2_history': list(state.co2_history),

                'bpm': state.latest_bpm,
                'hr_history': list(state.hr_history),

                'face_detected': face_detected,
                'face_emotion': face_emotion,
                'face_confidence': face_confidence,

                'plant_emotion': state.plant_emotion,
                'plant_confidence': state.plant_confidence,
                'plant_probs': state.plant_probs,
                'plant_model_is_mock': plant_model.is_mock,
            })
        except Exception:
            pass
        time.sleep(UPDATE_RATE_MS / 1000)


# ═══════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder='assets', static_url_path='/assets')
app.config['SECRET_KEY'] = 'phaenomena-live-dashboard-2026'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>LIVE DASHBOARD - Phänomena</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --dark: #201844; --panel-bg: rgba(40,30,80,0.6); --panel-border: rgba(255,255,255,0.06);
            --green: #1cffa8; --cyan: #2fd6ee; --pink: #ff6ac8; --amber: #f4c460;
            --text-dim: rgba(255,255,255,0.55); --heading: rgba(255,255,255,0.94);
        }
        html, body { height: 100%; overflow: hidden; }
        body {
            background: radial-gradient(circle at top, #2a2160 0%, var(--dark) 55%);
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .page {
            height: 100vh; display: flex; flex-direction: column;
            padding: 14px 26px 20px; gap: 10px;
        }

        .header-top { display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px; flex-shrink: 0; }
        .title-block .exhibit-title { font-size: 1.55em; font-weight: 800; letter-spacing: 0.04em; color: var(--green); }
        .title-block .exhibit-subtitle { font-size: 0.85em; color: var(--text-dim); margin-top: 2px; }

        .header-right { display: flex; flex-direction: column; align-items: flex-end; gap: 8px; flex-shrink: 0; }

        .brand-block { display: flex; align-items: center; gap: 10px; }
        .uni-logo { height: 32px; width: auto; object-fit: contain; display: block; }
        .uni-logo-fallback {
            height: 32px; width: 32px; border-radius: 7px; background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.28); align-items: center; justify-content: center;
            font-size: 0.62em; font-weight: 800; color: var(--heading); letter-spacing: 0.02em;
        }
        .brand-text { font-size: 1.05em; font-weight: 700; letter-spacing: 0.03em; color: var(--heading); }

        .status-bar { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; flex-shrink: 0; }
        .status-pill {
            display: flex; align-items: center; gap: 8px; padding: 6px 13px; border-radius: 20px;
            background: rgba(0,0,0,0.3); border: 1px solid var(--panel-border); font-size: 0.82em;
        }
        .status-dot { width: 9px; height: 9px; border-radius: 50%; background: #ef4444; flex-shrink: 0; }
        .status-dot.on { background: var(--green); box-shadow: 0 0 8px var(--green); }

        .top-grid {
            display: grid; grid-template-columns: repeat(3, 1fr); grid-template-rows: minmax(0, 1fr);
            gap: 16px; flex: 1.05 1 0; min-height: 0;
        }
        .charts-grid {
            display: grid; grid-template-columns: 2fr 1fr; grid-template-rows: minmax(0, 1fr);
            gap: 16px; flex: 0.95 1 0; min-height: 0;
        }
        @media (max-width: 1100px) { .top-grid, .charts-grid { grid-template-columns: 1fr; } }

        .panel {
            background: var(--panel-bg); border: 1px solid var(--panel-border); border-radius: 16px;
            padding: 14px 16px; display: flex; flex-direction: column; min-height: 0;
        }
        .panel-title {
            font-size: 0.78em; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase;
            color: var(--heading); margin-bottom: 8px; display: flex; justify-content: space-between;
            align-items: center; flex-shrink: 0;
        }
        .mock-badge {
            font-size: 0.75em; padding: 2px 9px; border-radius: 10px; background: rgba(244,196,96,0.18);
            color: var(--amber); border: 1px solid rgba(244,196,96,0.4); text-transform: none; letter-spacing: 0;
            font-weight: 600;
        }

        /* Camera panel */
        .camera-container { width: 100%; flex: 1; min-height: 0; border-radius: 12px; overflow: hidden; background: #000; }
        .camera-container img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .no-camera { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-dim); font-size: 0.85em; }
        .face-readout { margin-top: 8px; text-align: center; flex-shrink: 0; }
        .face-emotion { font-size: 1.2em; font-weight: 700; }
        .face-confidence { margin-top: 2px; font-size: 0.8em; color: var(--text-dim); }

        /* Plant emotion panel */
        .plant-emotion-label { font-size: 1.8em; font-weight: 800; text-align: center; margin: 4px 0 2px; text-transform: capitalize; flex-shrink: 0; }
        .plant-emotion-label.positiv { color: var(--green); }
        .plant-emotion-label.negativ { color: var(--pink); }
        .plant-emotion-label.neutral { color: var(--cyan); }
        .plant-confidence-text { text-align: center; color: var(--text-dim); font-size: 0.82em; margin-bottom: 10px; flex-shrink: 0; }
        .prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; font-size: 0.78em; flex-shrink: 0; }
        .prob-label { width: 62px; color: var(--text-dim); text-transform: capitalize; }
        .prob-bar-bg { flex: 1; height: 8px; border-radius: 4px; background: rgba(255,255,255,0.08); overflow: hidden; }
        .prob-bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
        .prob-bar-fill.negativ { background: var(--pink); }
        .prob-bar-fill.neutral { background: var(--cyan); }
        .prob-bar-fill.positiv { background: var(--green); }
        .prob-value { width: 34px; text-align: right; color: var(--text-dim); }
        .plant-quality-tag { margin-top: auto; padding-top: 8px; text-align: center; font-size: 0.75em; color: var(--text-dim); flex-shrink: 0; }

        /* Heart rate panel */
        .bpm-readout { text-align: center; margin-bottom: 6px; flex-shrink: 0; }
        .bpm-value { font-size: 2.2em; font-weight: 800; color: var(--pink); line-height: 1; }
        .bpm-unit { font-size: 0.8em; color: var(--text-dim); }

        canvas.chart { width: 100%; flex: 1; display: block; min-height: 0; }

        /* Wide chart panels */
        .chart-stat { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; flex-shrink: 0; }
        .chart-stat .value { font-size: 1.4em; font-weight: 700; }
        .chart-stat .value.plant { color: var(--green); }
        .chart-stat .value.co2 { color: var(--cyan); }
        .chart-stat .unit { font-size: 0.75em; color: var(--text-dim); margin-left: 4px; }

        .footer { flex-shrink: 0; text-align: center; font-size: 0.68em; color: var(--text-dim); line-height: 1.5; padding-top: 2px; }
        .footer .credits { color: rgba(255,255,255,0.6); }
        .footer .disclaimer { font-style: italic; opacity: 0.75; margin-top: 1px; }
    </style>
</head>
<body>
    <div class="page">
    <div class="header-top">
        <div class="title-block">
            <div class="exhibit-title">LIVE DASHBOARD</div>
            <div class="exhibit-subtitle">Plant · Camera · Heart · CO&#8322; &mdash; all live in real time</div>
        </div>
        <div class="header-right">
            <div class="brand-block">
                <img src="/assets/uni-koeln-logo.png" alt="University of Cologne" class="uni-logo" id="uniLogo"
                     onerror="this.style.display='none'; document.getElementById('uniLogoFallback').style.display='flex';">
                <div class="uni-logo-fallback" id="uniLogoFallback" style="display:none">UzK</div>
                <span class="brand-text">COIN 2026</span>
            </div>
            <div class="status-bar">
                <div class="status-pill"><div class="status-dot" id="dotPlant"></div>Plant</div>
                <div class="status-pill"><div class="status-dot" id="dotCamera"></div>Camera</div>
                <div class="status-pill"><div class="status-dot" id="dotPolar"></div>Polar H10</div>
            </div>
        </div>
    </div>

    <div class="top-grid">
        <!-- Camera + HSEmotion -->
        <div class="panel">
            <div class="panel-title">Camera &amp; Emotion Recognition (via HSEmotion)</div>
            <div class="camera-container" id="cameraContainer">
                <img src="/video_feed" id="cameraFeed" onerror="this.style.display='none'">
            </div>
            <div class="face-readout">
                <div class="face-emotion" id="faceEmotion">Looking for face...</div>
                <div class="face-confidence" id="faceConfidence">&nbsp;</div>
            </div>
        </div>

        <!-- Plant emotion (mock model) -->
        <div class="panel">
            <div class="panel-title">Plant Emotion (ML Prediction) <span class="mock-badge" id="mockBadge" style="display:none">MOCK MODEL</span></div>
            <div class="plant-emotion-label neutral" id="plantEmotionLabel">--</div>
            <div class="plant-confidence-text" id="plantConfidenceText">Collecting data...</div>
            <div id="probRows">
                <div class="prob-row"><div class="prob-label">Negative</div><div class="prob-bar-bg"><div class="prob-bar-fill negativ" id="probNegativ" style="width:0%"></div></div><div class="prob-value" id="probNegativVal">--</div></div>
                <div class="prob-row"><div class="prob-label">Neutral</div><div class="prob-bar-bg"><div class="prob-bar-fill neutral" id="probNeutral" style="width:0%"></div></div><div class="prob-value" id="probNeutralVal">--</div></div>
                <div class="prob-row"><div class="prob-label">Positive</div><div class="prob-bar-bg"><div class="prob-bar-fill positiv" id="probPositiv" style="width:0%"></div></div><div class="prob-value" id="probPositivVal">--</div></div>
            </div>
            <div class="plant-quality-tag" id="plantQualityTag">&nbsp;</div>
        </div>

        <!-- Heart rate -->
        <div class="panel">
            <div class="panel-title">Heart Rate (Polar H10)</div>
            <div class="bpm-readout">
                <span class="bpm-value" id="bpmValue">--</span>
                <span class="bpm-unit">bpm</span>
            </div>
            <canvas class="chart" id="hrCanvas"></canvas>
        </div>
    </div>

    <div class="charts-grid">
        <div class="panel chart-panel">
            <div class="panel-title">Plant Signal (Voltage)</div>
            <div class="chart-stat"><span class="value plant" id="voltageValue">--<span class="unit">mV</span></span></div>
            <canvas class="chart" id="plantCanvas"></canvas>
        </div>
        <div class="panel chart-panel">
            <div class="panel-title">CO&#8322;</div>
            <div class="chart-stat"><span class="value co2" id="co2Value">--<span class="unit">ppm</span></span></div>
            <canvas class="chart" id="co2Canvas"></canvas>
        </div>
    </div>

    <div class="footer">
        <div class="credits">COIN Seminar 2026 &middot; Supervisors: Peter Gloor, Simon Wolf &middot; Project: Plant Emotion Sensor &middot; Team 1: Nils Bringewatt, Jakob Jetter, Sarah Löhnert, Elisabeth Stein</div>
        <div class="disclaimer">University teaching project as part of the COIN Seminar &mdash; no data is stored.</div>
    </div>
    </div>

    <script>
        const socket = io();

        const faceLabels = {
            Happiness: 'Happiness 😊', Anger: 'Anger 😠', Contempt: 'Contempt 😒',
            Disgust: 'Disgust 🤢', Fear: 'Fear 😨', Sadness: 'Sadness 😢',
            Surprise: 'Surprise 😲', Neutral: 'Neutral 😐'
        };

        const plantLabels = { negativ: 'Negative', neutral: 'Neutral', positiv: 'Positive' };

        function drawLine(canvas, data, color) {
            const parent = canvas.parentElement;
            canvas.width = canvas.clientWidth || parent.clientWidth;
            canvas.height = canvas.clientHeight || 110;
            const ctx = canvas.getContext('2d');
            const w = canvas.width, h = canvas.height;
            ctx.clearRect(0, 0, w, h);
            if (!data || data.length < 2) return;
            const min = Math.min(...data), max = Math.max(...data);
            const range = (max - min) || 1, pad = range * 0.15;
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.8;
            ctx.lineJoin = 'round';
            for (let i = 0; i < data.length; i++) {
                const x = (i / (data.length - 1)) * w;
                const y = h - ((data[i] - min + pad) / (range + pad * 2)) * h;
                i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
            }
            ctx.stroke();
        }

        function pct(v) { return (v == null) ? '--' : Math.round(v * 100) + '%'; }

        socket.on('update', d => {
            document.getElementById('dotPlant').classList.toggle('on', d.plant_connected);
            document.getElementById('dotCamera').classList.toggle('on', d.camera_available);
            document.getElementById('dotPolar').classList.toggle('on', d.polar_connected);

            // Camera / face
            const cam = document.getElementById('cameraContainer');
            if (!d.camera_available) {
                cam.innerHTML = '<div class="no-camera">No camera connected</div>';
            }
            const faceEl = document.getElementById('faceEmotion');
            if (d.face_detected && d.face_emotion) {
                faceEl.textContent = faceLabels[d.face_emotion] || d.face_emotion;
                document.getElementById('faceConfidence').textContent = 'Confidence: ' + pct(d.face_confidence);
            } else {
                faceEl.textContent = 'Looking for face...';
                document.getElementById('faceConfidence').textContent = ' ';
            }

            // Plant emotion (mock model)
            document.getElementById('mockBadge').style.display = d.plant_model_is_mock ? 'inline-block' : 'none';
            const label = document.getElementById('plantEmotionLabel');
            if (d.plant_emotion) {
                label.textContent = plantLabels[d.plant_emotion] || d.plant_emotion;
                label.className = 'plant-emotion-label ' + d.plant_emotion;
                document.getElementById('plantConfidenceText').textContent = 'Confidence: ' + pct(d.plant_confidence);
            } else {
                label.textContent = '--';
                label.className = 'plant-emotion-label neutral';
                document.getElementById('plantConfidenceText').textContent = 'Collecting data...';
            }
            const probs = d.plant_probs || {};
            ['negativ', 'neutral', 'positiv'].forEach(cls => {
                const p = probs[cls] || 0;
                document.getElementById('prob' + cls.charAt(0).toUpperCase() + cls.slice(1)).style.width = (p * 100) + '%';
                document.getElementById('prob' + cls.charAt(0).toUpperCase() + cls.slice(1) + 'Val').textContent = pct(p);
            });
            document.getElementById('plantQualityTag').textContent = d.plant_quality ? ('Signal quality: ' + d.plant_quality) : ' ';

            // Heart rate
            document.getElementById('bpmValue').textContent = d.bpm != null ? d.bpm.toFixed(0) : '--';
            drawLine(document.getElementById('hrCanvas'), d.hr_history, '#ff6ac8');

            // Plant voltage
            document.getElementById('voltageValue').innerHTML = (d.voltages && d.voltages.length)
                ? d.voltages[d.voltages.length - 1].toFixed(1) + '<span class="unit">mV</span>' : '--';
            drawLine(document.getElementById('plantCanvas'), d.voltages, '#1cffa8');

            // CO2
            document.getElementById('co2Value').innerHTML = (d.co2 != null)
                ? d.co2.toFixed(0) + '<span class="unit">ppm</span>' : '--';
            drawLine(document.getElementById('co2Canvas'), d.co2_history, '#2fd6ee');
        });

        window.addEventListener('resize', () => {
            // canvases redraw on next 'update' tick; just trigger a resize-friendly repaint
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("PHAENOMENA - LIVE DASHBOARD")
    print("=" * 60)
    print("http://localhost:5004")
    print("=" * 60 + "\n")

    if camera.start():
        print("Camera started")
    else:
        print("Camera not available")

    threading.Thread(target=run_esp32_listener, daemon=True).start()
    threading.Thread(target=run_polar_listener, daemon=True).start()
    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=plant_prediction_loop, daemon=True).start()
    threading.Thread(target=broadcast_updates, daemon=True).start()

    socketio.run(app, host='0.0.0.0', port=5004, debug=False, allow_unsafe_werkzeug=True)
