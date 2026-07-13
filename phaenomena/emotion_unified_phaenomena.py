#!/usr/bin/env python3
"""
🌿 Phaenomena - EMOTIONSWAHRNEHMUNG (Emotion Perception)
Unified Exhibition Interface with Gesture Controls

Features:
- Phänomena Corporate Design (Town & Country title, Satoshi body)
- Gesture control for language selection (swipe left/right)
- Corner overlay camera for gesture detection
- Full-screen centered plant/emotion content
- 4-language support (DE/FR/IT/EN)

Port: 5003
Display: Optimized for 55" portrait monitor
"""

import asyncio
import json
import pickle
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime
import threading
import time
import wave
import cv2

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks, welch
from scipy.fft import fft, fftfreq

from flask import Flask, render_template_string, jsonify, Response, request, send_from_directory
from flask_socketio import SocketIO
import websockets

# Emotion detection libraries
USE_HSEMOTION = False
USE_HSEMOTION_ONNX = False
USE_DEEPFACE = False
MTCNN_DETECTOR = None
HSEMOTION_MODEL = None

print("\n🔍 Checking emotion recognition libraries...")
import sys
sys.stdout.flush()

try:
    print("   Trying HSEmotion-ONNX...", end=" ")
    sys.stdout.flush()
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer as HSEmotionONNX
    from facenet_pytorch import MTCNN
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    MTCNN_DETECTOR = MTCNN(keep_all=False, device=device, min_face_size=60)
    HSEMOTION_MODEL = HSEmotionONNX(model_name='enet_b0_8_best_afew')
    
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    _ = HSEMOTION_MODEL.predict_emotions(test_img, logits=False)
    
    USE_HSEMOTION_ONNX = True
    USE_HSEMOTION = True
    print("✅ OK")
except Exception as e:
    print(f"❌ {e}")

if not USE_HSEMOTION:
    try:
        print("   Trying HSEmotion...", end=" ")
        sys.stdout.flush()
        from hsemotion.facial_emotions import HSEmotionRecognizer
        from facenet_pytorch import MTCNN
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        MTCNN_DETECTOR = MTCNN(keep_all=False, device=device, min_face_size=60)
        HSEMOTION_MODEL = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device=device)
        
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _ = HSEMOTION_MODEL.predict_emotions(test_img, logits=False)
        
        USE_HSEMOTION = True
        print("✅ OK")
    except Exception as e:
        print(f"❌ {e}")
        MTCNN_DETECTOR = None
        HSEMOTION_MODEL = None

if not USE_HSEMOTION:
    try:
        print("   Trying DeepFace...", end=" ")
        sys.stdout.flush()
        from deepface import DeepFace
        USE_DEEPFACE = True
        print("✅ OK")
    except Exception as e:
        print(f"❌ {e}")

if not USE_HSEMOTION and not USE_DEEPFACE:
    print("\n⚠️  WARNING: No emotion recognition library available!")
else:
    lib_name = "HSEmotion-ONNX" if USE_HSEMOTION_ONNX else ("HSEmotion" if USE_HSEMOTION else "DeepFace")
    print(f"\n✅ Using {lib_name} for emotion recognition\n")

# ===== CONFIGURATION =====
ESP32_IP = ""
ESP32_PORT = 81

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

SAMPLE_RATE = 100
WINDOW_SECONDS = 10
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

WARMUP_PAIRS = 10
MAX_TRAINING_PAIRS = 200
RETRAIN_INTERVAL = 5
MIN_PAIRS_FOR_RETRAIN = 4

EMOTION_CONFIDENCE_THRESHOLD = 0.4
EMOTION_WINDOW_SECONDS = 2.0
MIN_SAMPLE_INTERVAL = 8.0
EMOTION_MAJORITY_THRESHOLD = 0.5

BUFFER_SECONDS = 2.0
FORWARD_COLLECTION_SECONDS = 8.0
TOTAL_COLLECTION_SECONDS = BUFFER_SECONDS + FORWARD_COLLECTION_SECONDS

POSITIVE_EMOTIONS = ['happy', 'surprise', 'happiness']
NEGATIVE_EMOTIONS = ['angry', 'sad', 'fear', 'disgust', 'anger', 'sadness', 'contempt']
NEUTRAL_EMOTIONS = ['neutral']

MODEL_MAX_AGE_MINUTES = 30
UPDATE_RATE_MS = 100
PREDICTION_INTERVAL = 0.5

OUTPUT_DIR = Path("./emotion_training_data")
OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_DIR = Path("./plant-viz-movies")
FONTS_DIR = Path("./fonts/Fonts")


def map_emotion_to_category(emotion):
    if emotion is None:
        return 'neutral'
    emotion_lower = emotion.lower()
    if emotion_lower in [e.lower() for e in POSITIVE_EMOTIONS]:
        return 'positive'
    elif emotion_lower in [e.lower() for e in NEGATIVE_EMOTIONS]:
        return 'negative'
    return 'neutral'


def map_emotion_with_confidence(emotion, confidence):
    category = map_emotion_to_category(emotion)
    if category == 'negative' and confidence < 0.3:
        return 'neutral', confidence
    return category, confidence


def save_wav_file(samples_v, label, sample_number):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"plant_emotion_{label}_n{sample_number:03d}_{timestamp}.wav"
    filepath = OUTPUT_DIR / filename
    
    samples_array = np.array(samples_v, dtype=np.float64)
    normalized = 2 * (samples_array - 0.5) / 2.5 - 1
    normalized = np.clip(normalized, -1, 1)
    int_samples = (normalized * 32767).astype(np.int16)
    
    with wave.open(str(filepath), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(int_samples.tobytes())
    return filepath


# ===== FEATURE EXTRACTION =====
def extract_features(samples):
    if len(samples) < 100:
        return None
    
    arr = np.array(samples)
    features = {}
    
    features['mean'] = np.mean(arr)
    features['std'] = np.std(arr)
    features['min'] = np.min(arr)
    features['max'] = np.max(arr)
    features['range'] = features['max'] - features['min']
    features['median'] = np.median(arr)
    features['iqr'] = np.percentile(arr, 75) - np.percentile(arr, 25)
    
    detrended = arr - np.linspace(arr[0], arr[-1], len(arr))
    features['detrend_std'] = np.std(detrended)
    
    zero_crossings = np.sum(np.abs(np.diff(np.sign(detrended - np.mean(detrended)))) > 0)
    features['zero_crossings'] = zero_crossings
    features['zero_crossing_rate'] = zero_crossings / len(arr)
    
    peaks_pos, _ = find_peaks(detrended, height=np.std(detrended) * 0.5)
    peaks_neg, _ = find_peaks(-detrended, height=np.std(detrended) * 0.5)
    features['num_peaks'] = len(peaks_pos) + len(peaks_neg)
    features['peak_rate'] = features['num_peaks'] / (len(arr) / SAMPLE_RATE)
    
    try:
        fft_vals = np.abs(fft(detrended))[:len(arr)//2]
        freqs = fftfreq(len(arr), 1/SAMPLE_RATE)[:len(arr)//2]
        
        if np.sum(fft_vals) > 0:
            features['spectral_centroid'] = np.sum(freqs * fft_vals) / np.sum(fft_vals)
        else:
            features['spectral_centroid'] = 0
        
        features['dominant_freq'] = freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0
        
        for low, high, name in [(0, 1, 'vlf'), (1, 5, 'lf'), (5, 15, 'mf'), (15, 50, 'hf')]:
            mask = (freqs >= low) & (freqs < high)
            features[f'power_{name}'] = np.sum(fft_vals[mask]) if np.any(mask) else 0
    except:
        features['spectral_centroid'] = 0
        features['dominant_freq'] = 0
        for name in ['vlf', 'lf', 'mf', 'hf']:
            features[f'power_{name}'] = 0
    
    try:
        diff1 = np.diff(arr)
        diff2 = np.diff(diff1)
        var0 = np.var(arr)
        var1 = np.var(diff1)
        var2 = np.var(diff2)
        
        features['hjorth_activity'] = var0
        features['hjorth_mobility'] = np.sqrt(var1 / var0) if var0 > 0 else 0
        features['hjorth_complexity'] = (np.sqrt(var2 / var1) / features['hjorth_mobility']) if var1 > 0 and features['hjorth_mobility'] > 0 else 0
    except:
        features['hjorth_activity'] = 0
        features['hjorth_mobility'] = 0
        features['hjorth_complexity'] = 0
    
    try:
        freqs_welch, psd = welch(arr, fs=SAMPLE_RATE, nperseg=min(256, len(arr)//4))
        features['psd_total'] = np.sum(psd)
        features['psd_peak_freq'] = freqs_welch[np.argmax(psd)] if len(psd) > 0 else 0
    except:
        features['psd_total'] = 0
        features['psd_peak_freq'] = 0
    
    return features


# ===== ML MODEL =====
class EmotionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_pairs = []
        self.samples_since_retrain = 0
        self.trained = False
        self.accuracy = 0.0
        self.model_path = OUTPUT_DIR / "emotion_model.pkl"
    
    def add_training_pair(self, features, label):
        self.training_pairs.append((features, label))
        self.samples_since_retrain += 1
        
        if len(self.training_pairs) > MAX_TRAINING_PAIRS:
            self.training_pairs = self.training_pairs[-MAX_TRAINING_PAIRS:]
        
        if self.samples_since_retrain >= RETRAIN_INTERVAL:
            self.train()
    
    def train(self):
        if len(self.training_pairs) < MIN_PAIRS_FOR_RETRAIN:
            return False
        
        labels = [lbl for _, lbl in self.training_pairs]
        pos_count = labels.count('positive')
        neg_count = labels.count('negative')
        
        if pos_count < 2 or neg_count < 2:
            return False
        
        X, y = [], []
        for features, label in self.training_pairs:
            if label in ['positive', 'negative']:
                X.append([features[k] for k in sorted(features.keys())])
                y.append(1 if label == 'positive' else 0)
        
        if len(X) < MIN_PAIRS_FOR_RETRAIN:
            return False
        
        X = np.array(X)
        y = np.array(y)
        self.feature_names = sorted(self.training_pairs[0][0].keys())
        
        try:
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=2, random_state=42)
            self.model.fit(X_scaled, y)
            self.accuracy = self.model.score(X_scaled, y)
            self.trained = True
            self.samples_since_retrain = 0
            print(f"✅ Model trained: accuracy={self.accuracy:.1%}, samples={len(X)}")
            self.save()
            return True
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
    
    def predict(self, features):
        if not self.trained or self.model is None:
            return None, 0.0
        try:
            X = np.array([[features[k] for k in self.feature_names]])
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)[0]
            prob = self.model.predict_proba(X_scaled)[0]
            return 'positive' if pred == 1 else 'negative', float(prob[pred])
        except:
            return None, 0.0
    
    def force_retrain(self):
        self.samples_since_retrain = RETRAIN_INTERVAL
        return self.train()
    
    def save(self):
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model, 'scaler': self.scaler, 'feature_names': self.feature_names,
                    'accuracy': self.accuracy, 'training_pairs': self.training_pairs[-50:],
                    'timestamp': datetime.now().isoformat()
                }, f)
        except Exception as e:
            print(f"❌ Save error: {e}")
    
    def load_latest(self):
        if not self.model_path.exists():
            return False, 0
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.accuracy = data['accuracy']
            self.training_pairs = data.get('training_pairs', [])
            self.trained = True
            age = (datetime.now() - datetime.fromisoformat(data['timestamp'])).total_seconds() / 60
            return True, age
        except:
            return False, 0


# ===== CAMERA HANDLER =====
class CameraHandler:
    def __init__(self):
        self.cap = None
        self.current_frame = None
        self.current_emotion = None
        self.current_category = None
        self.current_confidence = 0.0
        self.face_detected = False
        self.emotion_history = deque(maxlen=int(EMOTION_WINDOW_SECONDS * 10))
        self.stable_emotion = None
        self.lock = threading.Lock()
    
    def start(self):
        try:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            if self.cap.isOpened():
                ret, _ = self.cap.read()
                return ret
            return False
        except:
            return False
    
    def get_frame(self):
        with self.lock:
            if self.current_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    return jpeg.tobytes()
        return None
    
    def process_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        frame = cv2.flip(frame, 1)
        
        with self.lock:
            self.current_frame = frame.copy()
        
        if not USE_HSEMOTION and not USE_DEEPFACE:
            with self.lock:
                self.face_detected = False
            return
        
        try:
            if USE_HSEMOTION and MTCNN_DETECTOR is not None:
                self._detect_emotion_hsemotion(frame)
            elif USE_DEEPFACE:
                self._detect_emotion_deepface(frame)
            else:
                with self.lock:
                    self.face_detected = False
        except Exception as e:
            with self.lock:
                self.face_detected = False
    
    def _detect_emotion_hsemotion(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = MTCNN_DETECTOR.detect(rgb_frame)
            
            if boxes is None or len(boxes) == 0:
                with self.lock:
                    self.face_detected = False
                    self.current_emotion = None
                    self.current_category = None
                return
            
            box = boxes[0]
            x1, y1, x2, y2 = [int(b) for b in box]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return
            
            face_img = rgb_frame[y1:y2, x1:x2]
            if face_img.size == 0:
                return
            
            emotion, scores = HSEMOTION_MODEL.predict_emotions(face_img, logits=False)
            confidence = scores.get(emotion, 0.5) if isinstance(scores, dict) else 0.5
            category, adj_confidence = map_emotion_with_confidence(emotion, confidence)
            
            with self.lock:
                self.face_detected = True
                self.current_emotion = emotion
                self.current_category = category
                self.current_confidence = adj_confidence
                if category != 'neutral':
                    self.emotion_history.append((category, adj_confidence))
                self._check_stable_emotion()
        except Exception as e:
            pass
    
    def _detect_emotion_deepface(self, frame):
        if not USE_DEEPFACE:
            return
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            if results and len(results) > 0:
                result = results[0]
                emotions = result.get('emotion', {})
                if emotions:
                    dominant = result.get('dominant_emotion', max(emotions, key=emotions.get))
                    confidence = emotions.get(dominant, 0) / 100.0
                    category, adj_confidence = map_emotion_with_confidence(dominant, confidence)
                    
                    with self.lock:
                        self.face_detected = True
                        self.current_emotion = dominant
                        self.current_category = category
                        self.current_confidence = adj_confidence
                        if category != 'neutral':
                            self.emotion_history.append((category, adj_confidence))
                        self._check_stable_emotion()
        except:
            with self.lock:
                self.face_detected = False
    
    def _check_stable_emotion(self):
        if len(self.emotion_history) < 5:
            self.stable_emotion = None
            return
        
        recent = list(self.emotion_history)[-20:]
        categories = [cat for cat, _ in recent]
        pos_count = categories.count('positive')
        neg_count = categories.count('negative')
        total = len(categories)
        
        if pos_count / total >= EMOTION_MAJORITY_THRESHOLD:
            self.stable_emotion = 'positive'
        elif neg_count / total >= EMOTION_MAJORITY_THRESHOLD:
            self.stable_emotion = 'negative'
        else:
            self.stable_emotion = None


# ===== APPLICATION STATE =====
class AppState:
    def __init__(self):
        self.connected = False
        self.camera_connected = False
        self.mode = 'training'
        self.voltage_buffer = deque(maxlen=WINDOW_SIZE)
        self.total_samples = 0
        self.model = EmotionModel()
        self.training_phase = 'waiting'
        self.current_emotion = None
        self.collection_start_time = None
        self.collection_buffer = []
        self.last_sample_time = 0
        self.positive_count = 0
        self.negative_count = 0
        self.prediction = None
        self.prediction_confidence = 0.0
        self.last_prediction_time = 0
        self.emotion_delay_buffer = deque(maxlen=int(FORWARD_COLLECTION_SECONDS / 0.5))

state = AppState()
camera = CameraHandler()


# ===== ESP32 CONNECTION =====
async def esp32_listener():
    uri = f"ws://{ESP32_IP}:{ESP32_PORT}"
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                state.connected = True
                print(f"✅ Connected to ESP32")
                async for message in ws:
                    try:
                        data = json.loads(message)
                        for v in data.get('voltages', []):
                            voltage_v = v / 1000.0
                            state.voltage_buffer.append(voltage_v)
                            state.total_samples += 1
                            if state.training_phase == 'collecting':
                                state.collection_buffer.append(voltage_v)
                                if len(state.collection_buffer) >= int(TOTAL_COLLECTION_SECONDS * SAMPLE_RATE):
                                    process_collected_sample()
                    except:
                        pass
        except Exception as e:
            state.connected = False
            await asyncio.sleep(2)

def run_esp32_listener():
    asyncio.new_event_loop().run_until_complete(esp32_listener())


# ===== TRAINING LOGIC =====
def process_collected_sample():
    if state.current_emotion is None:
        state.training_phase = 'waiting'
        return
    
    samples = list(state.collection_buffer[:int(TOTAL_COLLECTION_SECONDS * SAMPLE_RATE)])
    features = extract_features(samples)
    
    if features is None:
        state.training_phase = 'waiting'
        state.collection_buffer = []
        return
    
    sample_num = state.positive_count + state.negative_count + 1
    save_wav_file(samples, state.current_emotion, sample_num)
    state.model.add_training_pair(features, state.current_emotion)
    
    if state.current_emotion == 'positive':
        state.positive_count += 1
    elif state.current_emotion == 'negative':
        state.negative_count += 1
    
    print(f"✅ Collected {state.current_emotion} sample ({state.positive_count}+/{state.negative_count}-)")
    
    state.training_phase = 'waiting'
    state.collection_buffer = []
    state.current_emotion = None
    state.last_sample_time = time.time()


def start_collection(category):
    if state.training_phase == 'collecting':
        return False
    
    current_time = time.time()
    if current_time - state.last_sample_time < MIN_SAMPLE_INTERVAL:
        return False
    
    buffer_samples = int(BUFFER_SECONDS * SAMPLE_RATE)
    state.collection_buffer = list(state.voltage_buffer)[-buffer_samples:]
    state.current_emotion = category
    state.training_phase = 'collecting'
    state.collection_start_time = current_time
    print(f"🎬 Recording {category} sample")
    return True


def camera_loop():
    while True:
        camera.process_frame()
        if state.mode == 'training' and state.training_phase == 'waiting':
            if camera.stable_emotion and camera.stable_emotion != 'neutral':
                if time.time() - state.last_sample_time >= MIN_SAMPLE_INTERVAL:
                    start_collection(camera.stable_emotion)
        time.sleep(0.1)


def prediction_loop():
    while True:
        if state.mode == 'exhibition' and state.model.trained:
            current_time = time.time()
            if current_time - state.last_prediction_time >= PREDICTION_INTERVAL:
                samples = list(state.voltage_buffer)
                if len(samples) >= WINDOW_SIZE:
                    features = extract_features(samples[-WINDOW_SIZE:])
                    if features:
                        pred, conf = state.model.predict(features)
                        state.prediction = pred
                        state.prediction_confidence = conf
                        state.last_prediction_time = current_time
        time.sleep(0.1)


def broadcast_updates():
    while True:
        try:
            with camera.lock:
                face_detected = camera.face_detected
                face_category = camera.current_category
                face_confidence = camera.current_confidence
            
            collection_progress = 0
            if state.training_phase == 'collecting':
                required = int(TOTAL_COLLECTION_SECONDS * SAMPLE_RATE)
                collection_progress = min(100, int(len(state.collection_buffer) / required * 100))
            
            socketio.emit('update', {
                'connected': state.connected,
                'mode': state.mode,
                'voltages': [v * 1000 for v in list(state.voltage_buffer)[-500:]],
                'face_detected': face_detected,
                'face_category': face_category,
                'face_confidence': face_confidence,
                'training_phase': state.training_phase,
                'current_emotion': state.current_emotion,
                'collection_progress': collection_progress,
                'positive_count': state.positive_count,
                'negative_count': state.negative_count,
                'model_trained': state.model.trained,
                'model_accuracy': state.model.accuracy,
                'prediction': state.prediction,
                'prediction_confidence': state.prediction_confidence,
                'warmup_pairs': WARMUP_PAIRS
            })
        except:
            pass
        time.sleep(UPDATE_RATE_MS / 1000)


# ===== FLASK APP =====
app = Flask(__name__)
app.config['SECRET_KEY'] = 'phaenomena-emotion-2026'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

import logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>EMOTIONSWAHRNEHMUNG - Phänomena</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
    <style>
        @font-face { font-family: 'TownCountry'; font-display: swap;
            src: url('/fonts/TownandCountryJNL.otf') format('opentype'),
                 url('/fonts/TownandCountryJNL-Regular.otf') format('opentype'),
                 url('/fonts/Town-and-Country-JNL.otf') format('opentype'),
                 url('/fonts/TownandCountryJNL.ttf') format('truetype'); }
        @font-face { font-family: 'Satoshi'; src: url('/fonts/Satoshi-Light.otf') format('opentype'); font-weight: 300; font-display: swap; }
        @font-face { font-family: 'Satoshi'; src: url('/fonts/Satoshi-Regular.otf') format('opentype'); font-weight: 400; font-display: swap; }
        @font-face { font-family: 'Satoshi'; src: url('/fonts/Satoshi-Medium.otf') format('opentype'); font-weight: 500; font-display: swap; }
        @font-face { font-family: 'Satoshi'; src: url('/fonts/Satoshi-Bold.otf') format('opentype'); font-weight: 700; font-display: swap; }
        @font-face { font-family: 'Satoshi'; src: url('/fonts/Satoshi-Black.otf') format('opentype'); font-weight: 900; font-display: swap; }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root { --dark: #201844; --green: #1cffa8; --cyan: #2fd6ee; --pink: #ff6ac8; --panel-bg: rgba(40,30,80,0.6); }
        
        body { background: var(--dark); color: #fff; font-family: 'Satoshi', sans-serif; height: 100vh; overflow: hidden; }
        
        /* ===== FULL-HEIGHT PORTRAIT LAYOUT ===== */
        .page-layout {
            display: flex;
            flex-direction: column;
            height: 100vh;
            padding: 0;
        }
        
        /* Padded content area */
        .content-area {
            display: flex;
            flex-direction: column;
            flex: 1;
            padding: 30px 40px 0 40px;
            min-height: 0;
        }
        
        /* Header */
        .header { text-align: center; margin-bottom: 30px; }
        .exhibit-title { font-family: 'TownCountry', serif; font-size: 3.2em; font-weight: 400; letter-spacing: 0.06em; color: var(--green); margin-bottom: 10px; }
        .exhibit-subtitle { font-size: 1.4em; font-weight: 400; color: rgba(255,255,255,0.85); }
        
        .connection-dot { position: fixed; top: 20px; right: 20px; width: 12px; height: 12px; border-radius: 50%; background: #ef4444; z-index: 100; }
        .connection-dot.connected { background: var(--green); box-shadow: 0 0 12px var(--green); }
        .connection-dot.face { right: 40px; }
        
        /* Main grid - two equal panels */
        .main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; flex: 1; min-height: 0; }
        
        .panel { background: var(--panel-bg); border-radius: 16px; padding: 24px; display: flex; flex-direction: column; align-items: center; }
        .panel-title { font-family: 'TownCountry', serif; font-size: 1.2em; font-weight: 400; letter-spacing: 0.04em; color: rgba(255,255,255,0.85); margin-bottom: 18px; text-align: center; }
        
        /* AI Panel - Camera */
        .camera-container { flex: 0 1 auto; width: 100%; max-width: 380px; aspect-ratio: 4/3; border-radius: 12px; overflow: hidden; background: #000; }
        .camera-container img { width: 100%; height: 100%; object-fit: cover; }
        
        .face-emotion { margin-top: 18px; padding: 12px 20px; border-radius: 12px; background: rgba(0,0,0,0.3); text-align: center; font-size: 1.3em; font-weight: 600; }
        .face-emotion.positive { color: var(--green); }
        .face-emotion.negative { color: var(--pink); }
        .face-emotion.neutral { color: rgba(255,255,255,0.6); }
        
        /* Plant Panel */
        .plant-content { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; gap: 12px; width: 100%; }
        
        .video-container { width: 300px; height: 300px; border-radius: 50%; overflow: hidden; border: 4px solid var(--green); box-shadow: 0 0 40px rgba(28,255,168,0.3); transition: all 0.4s; flex-shrink: 0; }
        .video-container.positive { border-color: var(--green); box-shadow: 0 0 60px rgba(28,255,168,0.5); }
        .video-container.negative { border-color: var(--pink); box-shadow: 0 0 60px rgba(255,106,200,0.5); }
        .video-container video { width: 100%; height: 100%; object-fit: cover; transform: scale(1.15); }
        
        .plant-sensing { font-size: 1.15em; color: rgba(255,255,255,0.7); margin-top: 12px; }
        .plant-response { font-size: 2em; font-weight: 700; text-align: center; }
        .plant-response.positive { color: var(--green); }
        .plant-response.negative { color: var(--pink); }
        .plant-response.neutral { color: rgba(255,255,255,0.5); }
        
        /* Confidence Scale - wider, clearer */
        .confidence-scale { width: 100%; max-width: 360px; margin-top: 18px; }
        .scale-labels { display: flex; justify-content: space-between; font-size: 0.8em; color: rgba(255,255,255,0.7); margin-bottom: 8px; }
        .scale-labels span { text-align: center; flex: 1; line-height: 1.3; }
        .scale-bar { height: 10px; background: linear-gradient(90deg, var(--pink) 0%, rgba(255,255,255,0.15) 50%, var(--green) 100%); border-radius: 5px; position: relative; }
        .scale-marker { position: absolute; top: -5px; width: 6px; height: 20px; background: white; border-radius: 3px; transform: translateX(-50%); transition: left 0.3s; box-shadow: 0 0 10px rgba(255,255,255,0.9); }
        .scale-ticks { display: flex; justify-content: space-between; margin-top: 5px; padding: 0 2px; }
        .scale-ticks span { width: 1px; height: 8px; background: rgba(255,255,255,0.3); }
        
        /* Middle section - training stuff */
        .middle-section { padding: 6px 40px; display: flex; flex-direction: column; gap: 8px; }
        
        /* Collection indicator */
        .collection-indicator { display: none; padding: 14px 24px; background: rgba(28,255,168,0.15); border: 2px solid var(--green); border-radius: 12px; text-align: center; animation: pulse-border 1s infinite; }
        .collection-indicator.active { display: block; }
        .collection-indicator.negative { background: rgba(255,106,200,0.15); border-color: var(--pink); }
        @keyframes pulse-border { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }
        .collection-indicator .progress-bar { width: 220px; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; margin: 8px auto 0; overflow: hidden; }
        .collection-indicator .progress-fill { height: 100%; background: var(--green); transition: width 0.2s; }
        
        /* Training stats */
        .training-stats { display: none; justify-content: center; gap: 30px; padding: 12px; background: rgba(0,0,0,0.2); border-radius: 12px; }
        .training-stats.visible { display: flex; }
        .stat-item { text-align: center; }
        .stat-value { font-size: 1.3em; font-weight: 700; }
        .stat-value.positive { color: var(--green); }
        .stat-value.negative { color: var(--pink); }
        .stat-label { font-size: 0.7em; color: rgba(255,255,255,0.5); }
        .training-progress { display: flex; align-items: center; gap: 8px; }
        .training-progress .progress-bar { width: 80px; height: 5px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden; }
        .training-progress .progress-fill { height: 100%; background: var(--green); }
        
        /* ===== SIGNAL - FULL WIDTH, NO PADDING ===== */
        .signal-container {
            width: 100%;
            height: 80px;
            background: rgba(255,255,255,0.03);
            border-top: 1px solid rgba(28,255,168,0.15);
            border-bottom: 1px solid rgba(28,255,168,0.15);
            padding: 4px 0;
            position: relative;
        }
        .signal-title {
            position: absolute; top: 6px; left: 16px;
            font-size: 0.65em; color: rgba(255,255,255,0.4);
            text-transform: uppercase; letter-spacing: 1.5px;
            z-index: 2;
        }
        #signalCanvas { width: 100%; height: 100%; display: block; }
        
        /* ===== BOTTOM AREA: Stick-man + language buttons ===== */
        .bottom-area {
            display: flex;
            align-items: flex-start;
            justify-content: flex-start;
            padding: 10px 40px 12px 40px;
            position: relative;
            min-height: 160px;
        }
        
        /* Language zone: stick-man on top, buttons below, centered */
        .language-zone {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }
        
        /* Gesture camera/avatar: same width as buttons row */
        .gesture-overlay {
            width: 280px;
            height: 190px;
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid rgba(28,255,168,0.3);
            background: rgba(0,0,0,0.5);
            transition: all 0.3s;
        }
        .gesture-overlay:hover { border-color: var(--green); }
        .gesture-overlay.detected { border-color: var(--green); box-shadow: 0 0 15px rgba(28,255,168,0.3); }
        #gestureVideo { display: none; }
        #gestureCanvas { width: 100%; height: 100%; }
        
        .gesture-feedback {
            text-align: center; font-size: 1.3em; font-weight: 700;
            color: var(--green); text-shadow: 0 0 15px var(--green);
            opacity: 0; pointer-events: none; height: 0; overflow: visible;
        }
        .gesture-feedback.active { opacity: 1; animation: pop 0.5s ease-out; }
        @keyframes pop { 0% { transform: scale(0.5); opacity: 0; } 50% { transform: scale(1.1); opacity: 1; } 100% { transform: scale(1); opacity: 1; } }
        
        .swipe-hint { text-align: center; font-size: 0.7em; color: rgba(255,255,255,0.35); margin-top: 2px; }
        
        .language-buttons { display: flex; gap: 8px; }
        .lang-btn {
            padding: 10px 20px; border-radius: 18px; font-size: 1em; font-weight: 600;
            background: rgba(32,24,68,0.9); border: 2px solid rgba(255,255,255,0.2);
            color: rgba(255,255,255,0.5); cursor: pointer; transition: all 0.3s;
            font-family: 'Satoshi', sans-serif; min-width: 58px; text-align: center;
        }
        .lang-btn:hover { border-color: rgba(255,255,255,0.4); color: rgba(255,255,255,0.8); }
        .lang-btn.active { background: var(--green); border-color: var(--green); color: var(--dark); }
        
        /* ===== QR PANEL - BOTTOM RIGHT, COMPACT ===== */
        .qr-panel {
            position: fixed; bottom: 14px; right: 14px;
            padding: 8px; border-radius: 10px;
            background: rgba(0,0,0,0.4); border: 1px solid rgba(28,255,168,0.25);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            width: 110px; z-index: 110;
        }
        .qr-panel h4 { margin: 0 0 4px; font-size: 0.65em; letter-spacing: 0.04em; color: var(--green); font-family: 'Satoshi', sans-serif; }
        .qr-label { margin-top: 3px; font-size: 0.45em; color: rgba(255,255,255,0.6); word-break: break-all; }
        .qr-hint { margin-top: 2px; font-size: 0.45em; color: rgba(255,255,255,0.4); }
        
        /* Staff controls - subtle, above QR */
        .staff-controls {
            position: fixed; bottom: 10px; right: 140px;
            display: flex; gap: 4px;
            opacity: 0.15; transition: opacity 0.3s; z-index: 100;
        }
        .staff-controls:hover { opacity: 1; }
        .staff-btn {
            background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
            color: rgba(255,255,255,0.7); padding: 5px 10px; border-radius: 5px;
            cursor: pointer; font-family: inherit; font-size: 0.65em;
        }
        .staff-btn:hover { background: rgba(255,255,255,0.2); }
        .staff-btn.active { background: rgba(28,255,168,0.3); border-color: var(--green); color: var(--green); }
        .staff-btn.positive { background: rgba(28,255,168,0.3); border-color: var(--green); color: var(--green); }
        .staff-btn.negative { background: rgba(255,106,200,0.3); border-color: var(--pink); color: var(--pink); }
        
        .no-camera { display: flex; align-items: center; justify-content: center; height: 100%; font-size: 0.65em; color: rgba(255,255,255,0.4); }
    </style>
</head>
<body>
    <div class="connection-dot" id="plantDot"></div>
    <div class="connection-dot face" id="cameraDot"></div>
    
    <div class="page-layout">
        <!-- Padded content -->
        <div class="content-area">
            <div class="header">
                <div class="exhibit-title" id="exhibitTitle">EMOTIONSWAHRNEHMUNG</div>
                <div class="exhibit-subtitle" id="exhibitSubtitle">Spüre ich deine Emotionen? Probiere es aus.</div>
            </div>
            
            <div class="main-grid">
                <!-- AI Panel -->
                <div class="panel">
                    <div class="panel-title" id="aiPanelTitle">Das erkennt die KI:</div>
                    <div class="camera-container">
                        <img src="/video_feed" id="cameraFeed">
                    </div>
                    <div class="face-emotion neutral" id="faceEmotion">Suche Gesicht...</div>
                </div>
                
                <!-- Plant Panel -->
                <div class="panel">
                    <div class="panel-title" id="plantPanelTitle">Das spürt die Pflanze:</div>
                    <div class="plant-content">
                        <div class="video-container" id="videoContainer">
                            <video id="plantVideo" autoplay loop muted playsinline>
                                <source src="/videos/happy.mp4" type="video/mp4">
                            </video>
                        </div>
                        <div class="plant-sensing" id="plantSensing">Ich spüre...</div>
                        <div class="plant-response neutral" id="plantResponse">--</div>
                        
                        <div class="confidence-scale">
                            <div class="scale-labels">
                                <span id="scaleLabel1">Trauer<br>sicher</span>
                                <span id="scaleLabel2">Trauer<br>wahrscheinlich</span>
                                <span id="scaleLabel3">Freude<br>wahrscheinlich</span>
                                <span id="scaleLabel4">Freude<br>sicher</span>
                            </div>
                            <div class="scale-bar">
                                <div class="scale-marker" id="scaleMarker" style="left:50%"></div>
                            </div>
                            <div class="scale-ticks"><span></span><span></span><span></span><span></span><span></span></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Training stuff (padded) -->
        <div class="middle-section">
            <div class="collection-indicator" id="collectionIndicator">
                <span id="recordingText">Aufnahme läuft...</span>
                <div class="progress-bar"><div class="progress-fill" id="collectionProgress" style="width:0%"></div></div>
            </div>
            
            <div class="training-stats" id="trainingStats">
                <div class="stat-item"><div class="stat-value positive" id="positiveCount">0</div><div class="stat-label">😊 Positiv</div></div>
                <div class="stat-item"><div class="stat-value negative" id="negativeCount">0</div><div class="stat-label">😢 Negativ</div></div>
                <div class="training-progress"><div class="progress-bar"><div class="progress-fill" id="warmupProgress" style="width:0%"></div></div><div class="stat-label" id="warmupText">0/10</div></div>
            </div>
        </div>
        
        <!-- Signal: FULL WIDTH edge-to-edge -->
        <div class="signal-container">
            <div class="signal-title" id="signalTitle">Elektromagnetisches Feld der Pflanze</div>
            <canvas id="signalCanvas"></canvas>
        </div>
        
        <!-- Bottom: Stick-man on top of language buttons, centered -->
        <div class="bottom-area">
            <div class="language-zone">
                <div class="gesture-overlay" id="gestureOverlay">
                    <video id="gestureVideo" playsinline></video>
                    <canvas id="gestureCanvas"></canvas>
                </div>
                <div class="gesture-feedback" id="gestureFeedback"></div>
                <div class="swipe-hint" id="swipeHint">👋 Wische für Sprache</div>
                <div class="language-buttons">
                    <button class="lang-btn active" data-lang="de">DE</button>
                    <button class="lang-btn" data-lang="en">EN</button>
                    <button class="lang-btn" data-lang="fr">FR</button>
                    <button class="lang-btn" data-lang="it">IT</button>
                </div>
            </div>
        </div>
    </div>
    
    <!-- QR code - bottom right -->
    <div class="qr-panel">
        <h4>Plantangle</h4>
        <div id="qrBox"></div>
        <div class="qr-label" id="qrLabel"></div>
        <div class="qr-hint">Scan for sensor IP</div>
    </div>
    
    <!-- Staff controls - bottom right, above QR -->
    <div class="staff-controls">
        <button class="staff-btn positive" onclick="recordEmotion('positive')">😊+</button>
        <button class="staff-btn negative" onclick="recordEmotion('negative')">😢-</button>
        <button class="staff-btn active" id="btnTraining" onclick="setMode('training')">Train</button>
        <button class="staff-btn" id="btnExhibition" onclick="setMode('exhibition')">Exhibit</button>
        <button class="staff-btn" onclick="retrain()">Reset</button>
    </div>
    
    <script>
        const plantWs = "{{ plant_ws }}";
        const socket = io();
        let currentLang = 'de', currentMode = 'training';
        const languages = ['de', 'en', 'fr', 'it'];
        
        const T = {
            de: { 
                title: 'EMOTIONSWAHRNEHMUNG', 
                subtitle: 'Spüre ich deine Emotionen? Probiere es aus.',
                aiPanel: 'Das erkennt die KI:',
                plantPanel: 'Das spürt die Pflanze:',
                lookingForFace: 'Suche Gesicht...', 
                sensing: 'Ich spüre...', 
                recording: 'Aufnahme läuft...', 
                signalTitle: 'Elektromagnetisches Feld der Pflanze', 
                swipeHint: '👋 Wische für Sprache',
                scale: ['Trauer<br>sicher', 'Trauer<br>wahrscheinlich', 'Freude<br>wahrscheinlich', 'Freude<br>sicher'],
                face: { positive: '😊 Freude erkannt', negative: '😢 Trauer erkannt', neutral: '😐 Neutral' }, 
                plant: { positive: 'Freude', negative: 'Trauer', neutral: '--' }
            },
            en: { 
                title: 'EMOTION PERCEPTION', 
                subtitle: 'Can I feel your emotions? Try it out.',
                aiPanel: 'AI recognizes:',
                plantPanel: 'The plant senses:',
                lookingForFace: 'Looking for face...', 
                sensing: 'I sense...', 
                recording: 'Recording...', 
                signalTitle: 'Electromagnetic field of the plant', 
                swipeHint: '👋 Swipe for language',
                scale: ['Sadness<br>certain', 'Sadness<br>probable', 'Happiness<br>probable', 'Happiness<br>certain'],
                face: { positive: '😊 Joy detected', negative: '😢 Sadness detected', neutral: '😐 Neutral' }, 
                plant: { positive: 'Happiness', negative: 'Sadness', neutral: '--' }
            },
            fr: { 
                title: 'PERCEPTION DES ÉMOTIONS', 
                subtitle: 'Est-ce que je ressens vos émotions? Essayez.',
                aiPanel: "L'IA reconnaît:",
                plantPanel: 'La plante ressent:',
                lookingForFace: 'Recherche de visage...', 
                sensing: 'Je ressens...', 
                recording: 'Enregistrement...', 
                signalTitle: 'Champ électromagnétique de la plante', 
                swipeHint: '👋 Glissez pour langue',
                scale: ['Tristesse<br>certaine', 'Tristesse<br>probable', 'Joie<br>probable', 'Joie<br>certaine'],
                face: { positive: '😊 Joie détectée', negative: '😢 Tristesse détectée', neutral: '😐 Neutre' }, 
                plant: { positive: 'Joie', negative: 'Tristesse', neutral: '--' }
            },
            it: { 
                title: 'PERCEZIONE DELLE EMOZIONI', 
                subtitle: 'Posso sentire le tue emozioni? Provaci.',
                aiPanel: "L'IA riconosce:",
                plantPanel: 'La pianta sente:',
                lookingForFace: 'Cercando viso...', 
                sensing: 'Sento...', 
                recording: 'Registrazione...', 
                signalTitle: 'Campo elettromagnetico della pianta', 
                swipeHint: '👋 Scorri per lingua',
                scale: ['Tristezza<br>certa', 'Tristezza<br>probabile', 'Felicità<br>probabile', 'Felicità<br>certa'],
                face: { positive: '😊 Gioia rilevata', negative: '😢 Tristezza rilevata', neutral: '😐 Neutrale' }, 
                plant: { positive: 'Felicità', negative: 'Tristezza', neutral: '--' }
            }
        };
        
        const t = () => T[currentLang] || T.en;
        
        function updateLang(lang) {
            currentLang = lang;
            document.querySelectorAll('.lang-btn').forEach(b => b.classList.toggle('active', b.dataset.lang === lang));
            const tr = t();
            document.getElementById('exhibitTitle').textContent = tr.title;
            document.getElementById('exhibitSubtitle').textContent = tr.subtitle;
            document.getElementById('aiPanelTitle').textContent = tr.aiPanel;
            document.getElementById('plantPanelTitle').textContent = tr.plantPanel;
            document.getElementById('plantSensing').textContent = tr.sensing;
            document.getElementById('signalTitle').textContent = tr.signalTitle;
            document.getElementById('swipeHint').textContent = tr.swipeHint;
            document.getElementById('recordingText').textContent = tr.recording;
            document.getElementById('scaleLabel1').innerHTML = tr.scale[0];
            document.getElementById('scaleLabel2').innerHTML = tr.scale[1];
            document.getElementById('scaleLabel3').innerHTML = tr.scale[2];
            document.getElementById('scaleLabel4').innerHTML = tr.scale[3];
        }
        
        function cycleLang(dir) {
            const idx = languages.indexOf(currentLang);
            updateLang(languages[(idx + dir + 4) % 4]);
            showFeedback('🌐 ' + currentLang.toUpperCase());
        }
        
        document.querySelectorAll('.lang-btn').forEach(btn => btn.addEventListener('click', () => updateLang(btn.dataset.lang)));
        document.addEventListener('keydown', e => {
            if (e.key >= '1' && e.key <= '4') updateLang(languages[e.key - 1]);
            if (e.key === 'ArrowLeft') cycleLang(-1);
            if (e.key === 'ArrowRight') cycleLang(1);
        });
        
        // Gesture control
        const gestureVideo = document.getElementById('gestureVideo');
        const gestureCanvas = document.getElementById('gestureCanvas');
        const gestureCtx = gestureCanvas.getContext('2d');
        const gestureOverlay = document.getElementById('gestureOverlay');
        
        let handHistory = { left: [], right: [] }, langCooldown = false, animPhase = 0;
        const swipeThreshold = 1.2, histLen = 8;
        const colors = { stem: '#2d6b2d', leaf: '#3d8b3d', flower: '#e8b4d8', center: '#f4d03f' };
        
        async function initGesture() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
                gestureVideo.srcObject = stream;
                await gestureVideo.play();
                gestureCanvas.width = gestureVideo.videoWidth;
                gestureCanvas.height = gestureVideo.videoHeight;
                
                const pose = new Pose({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}` });
                pose.setOptions({ modelComplexity: 0, smoothLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
                pose.onResults(onPose);
                
                (async function loop() { await pose.send({ image: gestureVideo }); requestAnimationFrame(loop); })();
            } catch (e) {
                gestureOverlay.innerHTML = '<div class="no-camera">📷</div>';
            }
        }
        
        function onPose(results) {
            const w = gestureCanvas.width, h = gestureCanvas.height;
            animPhase += 0.03;
            gestureCtx.save();
            gestureCtx.clearRect(0, 0, w, h);
            gestureCtx.translate(w, 0);
            gestureCtx.scale(-1, 1);
            gestureCtx.fillStyle = '#0a0f14';
            gestureCtx.fillRect(0, 0, w, h);
            
            let detected = false;
            if (results.poseLandmarks) {
                const lm = results.poseLandmarks;
                const vis = (lm[11].visibility + lm[12].visibility) / 2;
                if (vis > 0.5) { detected = true; drawAvatar(lm, w, h); detectGestures(lm); }
            }
            gestureOverlay.classList.toggle('detected', detected);
            gestureCtx.restore();
        }
        
        function drawAvatar(lm, w, h) {
            const [nose, lSh, rSh, lEl, rEl, lWr, rWr, lHip, rHip, lKn, rKn, lAn, rAn] = [0,11,12,13,14,15,16,23,24,25,26,27,28].map(i => lm[i]);
            const shC = { x: (lSh.x + rSh.x) / 2 * w, y: (lSh.y + rSh.y) / 2 * h };
            const hiC = { x: (lHip.x + rHip.x) / 2 * w, y: (lHip.y + rHip.y) / 2 * h };
            
            gestureCtx.strokeStyle = colors.stem; gestureCtx.lineCap = 'round';
            gestureCtx.lineWidth = 8;
            [[lAn, lKn, lHip], [rAn, rKn, rHip]].forEach(([a, k, hp]) => {
                gestureCtx.beginPath(); gestureCtx.moveTo(a.x*w, a.y*h); gestureCtx.quadraticCurveTo(k.x*w, k.y*h, hp.x*w, hp.y*h); gestureCtx.stroke();
            });
            
            const mx = (hiC.x + shC.x) / 2 + Math.sin(animPhase) * 2, my = (hiC.y + shC.y) / 2;
            gestureCtx.lineWidth = 10; gestureCtx.beginPath(); gestureCtx.moveTo(hiC.x, hiC.y); gestureCtx.quadraticCurveTo(mx, my, shC.x, shC.y); gestureCtx.stroke();
            drawLeaf(mx - 8, my - 10, -0.8, 14); drawLeaf(mx + 8, my + 6, 0.8, 12);
            
            gestureCtx.lineWidth = 6;
            [[lSh, lEl, lWr, true], [rSh, rEl, rWr, false]].forEach(([sh, el, wr, isL]) => {
                gestureCtx.beginPath(); gestureCtx.moveTo(sh.x*w, sh.y*h); gestureCtx.lineTo(el.x*w, el.y*h); gestureCtx.lineTo(wr.x*w, wr.y*h); gestureCtx.stroke();
                drawLeaf(el.x*w, el.y*h, isL ? -0.6 : 0.6, 12);
                for (let i = -1; i <= 1; i++) drawLeaf(wr.x*w, wr.y*h, (isL ? Math.PI : 0) + i * 0.4, 10);
            });
            
            const hx = nose.x * w, hy = nose.y * h, r = 24;
            gestureCtx.lineWidth = 7; gestureCtx.beginPath(); gestureCtx.moveTo(shC.x, shC.y); gestureCtx.lineTo(hx, hy + r * 0.3); gestureCtx.stroke();
            for (let i = 0; i < 8; i++) {
                gestureCtx.save(); gestureCtx.translate(hx, hy); gestureCtx.rotate((i / 8) * Math.PI * 2 + animPhase * 0.04);
                gestureCtx.fillStyle = colors.flower; gestureCtx.beginPath(); gestureCtx.ellipse(0, -r * 0.4, r * 0.28, r * 0.45, 0, 0, Math.PI * 2); gestureCtx.fill(); gestureCtx.restore();
            }
            gestureCtx.fillStyle = colors.center; gestureCtx.beginPath(); gestureCtx.arc(hx, hy, r * 0.28, 0, Math.PI * 2); gestureCtx.fill();
        }
        
        function drawLeaf(x, y, ang, sz) {
            gestureCtx.save(); gestureCtx.translate(x, y); gestureCtx.rotate(ang + Math.sin(animPhase + x * 0.02) * 0.05);
            gestureCtx.fillStyle = colors.leaf; gestureCtx.beginPath(); gestureCtx.moveTo(0, 0);
            gestureCtx.quadraticCurveTo(sz * 0.5, -sz * 0.26, sz, 0); gestureCtx.quadraticCurveTo(sz * 0.5, sz * 0.26, 0, 0);
            gestureCtx.fill(); gestureCtx.restore();
        }
        
        function detectGestures(lm) {
            const now = Date.now(), lWr = lm[15], rWr = lm[16];
            handHistory.left.push({ x: lWr.x, y: lWr.y, t: now }); handHistory.right.push({ x: rWr.x, y: rWr.y, t: now });
            if (handHistory.left.length > histLen) handHistory.left.shift();
            if (handHistory.right.length > histLen) handHistory.right.shift();
            if (handHistory.left.length < 3) return;
            
            const lV = calcVel(handHistory.left), rV = calcVel(handHistory.right);
            const lSwR = lV.x < -swipeThreshold && Math.abs(lV.y) < 0.5;
            const lSwL = lV.x > swipeThreshold && Math.abs(lV.y) < 0.5;
            const rSwR = rV.x < -swipeThreshold && Math.abs(rV.y) < 0.5;
            const rSwL = rV.x > swipeThreshold && Math.abs(rV.y) < 0.5;
            const rStill = Math.abs(rV.x) < 0.4, lStill = Math.abs(lV.x) < 0.4;
            
            if (!langCooldown) {
                // Only right-to-left swipe cycles language forward (less fickle)
                if ((lSwR && rStill) || (rSwR && lStill)) { cycleLang(1); langCooldown = true; handHistory = { left: [], right: [] }; setTimeout(() => langCooldown = false, 1000); }
            }
        }
        
        function calcVel(h) {
            if (h.length < 2) return { x: 0, y: 0 };
            let vx = 0, vy = 0, tw = 0;
            for (let i = 1; i < h.length; i++) {
                const dt = (h[i].t - h[i-1].t) / 1000;
                if (dt > 0 && dt < 0.5) { vx += ((h[i].x - h[i-1].x) / dt) * i; vy += ((h[i].y - h[i-1].y) / dt) * i; tw += i; }
            }
            return tw ? { x: vx / tw, y: vy / tw } : { x: 0, y: 0 };
        }
        
        function showFeedback(txt) {
            const el = document.getElementById('gestureFeedback');
            el.textContent = txt; el.classList.remove('active'); void el.offsetWidth; el.classList.add('active');
            setTimeout(() => el.classList.remove('active'), 500);
        }
        
        // Signal canvas - FULL WIDTH
        const sigCanvas = document.getElementById('signalCanvas'), sigCtx = sigCanvas.getContext('2d');
        function resizeSig() {
            const c = sigCanvas.parentElement;
            sigCanvas.width = c.clientWidth;
            sigCanvas.height = c.clientHeight - 12;
        }
        function drawSignal(data, color) {
            resizeSig(); const w = sigCanvas.width, h = sigCanvas.height; sigCtx.clearRect(0, 0, w, h);
            if (data.length < 2) return;
            const min = Math.min(...data), max = Math.max(...data), range = max - min || 1, pad = range * 0.1;
            sigCtx.beginPath(); sigCtx.strokeStyle = color; sigCtx.lineWidth = 1.5;
            for (let i = 0; i < data.length; i++) { const x = (i / data.length) * w, y = h - ((data[i] - min + pad) / (range + pad * 2)) * h; i ? sigCtx.lineTo(x, y) : sigCtx.moveTo(x, y); }
            sigCtx.stroke();
        }
        
        // Video
        const plantVideo = document.getElementById('plantVideo'); let vidState = 'happy';
        function switchVideo(s) { if (vidState !== s) { plantVideo.src = `/videos/${s}.mp4`; plantVideo.load(); plantVideo.play().catch(() => {}); vidState = s; } }
        
        // Mode
        function updateMode(mode) {
            currentMode = mode;
            document.getElementById('btnTraining').classList.toggle('active', mode === 'training');
            document.getElementById('btnExhibition').classList.toggle('active', mode === 'exhibition');
            document.getElementById('trainingStats').classList.toggle('visible', mode === 'training');
        }
        function setMode(mode) { if (mode !== currentMode) fetch('/toggle_mode', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode }) }).then(r => r.json()).then(d => updateMode(d.mode)); }
        function retrain() { fetch('/retrain', { method: 'POST' }); }
        function recordEmotion(emo) { fetch('/record_emotion', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ emotion: emo }) }); }
        
        // Socket
        socket.on('update', d => {
            document.getElementById('plantDot').classList.toggle('connected', d.connected);
            document.getElementById('cameraDot').classList.toggle('connected', d.face_detected);
            
            if (d.mode !== currentMode) updateMode(d.mode);
            
            const faceEl = document.getElementById('faceEmotion');
            const tr = t();
            if (d.face_detected && d.face_category) {
                faceEl.textContent = tr.face[d.face_category] || tr.face.neutral;
                faceEl.className = 'face-emotion ' + d.face_category;
            } else {
                faceEl.textContent = tr.lookingForFace;
                faceEl.className = 'face-emotion neutral';
            }
            
            const vc = document.getElementById('videoContainer');
            const pr = document.getElementById('plantResponse');
            const sm = document.getElementById('scaleMarker');
            
            if (d.mode === 'exhibition' && d.prediction) {
                const pred = d.prediction, conf = d.prediction_confidence || 0.5;
                switchVideo(pred === 'positive' ? 'happy' : 'sad');
                vc.className = 'video-container ' + pred;
                pr.textContent = tr.plant[pred] || tr.plant.neutral;
                pr.className = 'plant-response ' + pred;
                const scalePos = pred === 'positive' ? 50 + (conf * 50) : 50 - (conf * 50);
                sm.style.left = scalePos + '%';
            } else {
                pr.textContent = tr.plant.neutral;
                pr.className = 'plant-response neutral';
                sm.style.left = '50%';
            }
            
            document.getElementById('positiveCount').textContent = d.positive_count || 0;
            document.getElementById('negativeCount').textContent = d.negative_count || 0;
            const total = (d.positive_count || 0) + (d.negative_count || 0);
            document.getElementById('warmupProgress').style.width = Math.min(100, total / d.warmup_pairs * 100) + '%';
            document.getElementById('warmupText').textContent = `${total}/${d.warmup_pairs}`;
            
            const coll = document.getElementById('collectionIndicator');
            if (d.training_phase === 'collecting') {
                coll.classList.add('active');
                coll.classList.toggle('negative', d.current_emotion === 'negative');
                document.getElementById('collectionProgress').style.width = (d.collection_progress || 0) + '%';
            } else {
                coll.classList.remove('active');
            }
            
            if (d.voltages?.length > 0) {
                const color = (d.mode === 'exhibition' && d.prediction === 'negative') ? '#ff6ac8' : '#1cffa8';
                drawSignal(d.voltages, color);
            }
        });
        
        // Build QR for quick connect
        if (window.QRCode) {
            new QRCode(document.getElementById('qrBox'), {
                text: plantWs,
                width: 80,
                height: 80,
                colorDark: "#1cffa8",
                colorLight: "#0a0f14",
                correctLevel: QRCode.CorrectLevel.M
            });
            document.getElementById('qrLabel').textContent = plantWs;
        }

        window.addEventListener('resize',resizeSig);
        setTimeout(()=>{ resizeSig(); }, 100);
        setTimeout(()=>{ resizeSig(); }, 500);
        initGesture();
        document.addEventListener('click', () => plantVideo.play().catch(() => {}), { once: true });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, plant_ws=f"ws://{ESP32_IP}:{ESP32_PORT}")

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(VIDEO_DIR, filename)

@app.route('/fonts/debug')
def debug_fonts():
    """List all font files found in font directories for debugging"""
    result = {'FONTS_DIR': str(FONTS_DIR), 'exists': FONTS_DIR.exists(), 'files': {}}
    if FONTS_DIR.exists():
        for p in sorted(FONTS_DIR.rglob('*')):
            if p.is_file():
                result['files'][str(p.relative_to(FONTS_DIR))] = p.stat().st_size
    for alt in [Path("./Fonts"), Path("./fonts"), Path(".")]:
        if alt.exists():
            for ext in ['*.otf', '*.ttf', '*.woff', '*.woff2']:
                for p in sorted(alt.rglob(ext)):
                    result['files'][f"[{alt}] {p.relative_to(alt)}"] = p.stat().st_size
    return jsonify(result)

@app.route('/fonts/<path:filename>')
def serve_font(filename):
    fname_lower = filename.lower()
    search_dirs = [
        FONTS_DIR / "Satoshi Fonts",
        FONTS_DIR / "TownandCountryJNL",
        FONTS_DIR,
        Path("./Fonts/Satoshi Fonts"),
        Path("./Fonts/TownandCountryJNL"),
        Path("./Fonts"),
        Path("./fonts"),
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        exact = d / filename
        if exact.exists():
            return send_from_directory(d, filename)
        for f in d.iterdir():
            if f.is_file() and f.name.lower() == fname_lower:
                return send_from_directory(d, f.name)
            if f.is_file() and f.suffix.lower() in ['.otf', '.ttf', '.woff', '.woff2']:
                if fname_lower.replace('-', '').replace(' ', '').split('.')[0] in f.name.lower().replace('-', '').replace(' ', ''):
                    return send_from_directory(d, f.name)
    print(f"⚠️  Font not found: {filename} (searched {[str(d) for d in search_dirs if d.exists()]})")
    return "Font not found", 404

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response

@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    data = request.get_json(silent=True) or {}
    requested_mode = data.get('mode')
    if requested_mode == 'exhibition' and state.model.trained:
        state.mode = 'exhibition'
    elif requested_mode == 'training':
        state.mode = 'training'
    return jsonify({'mode': state.mode})

@app.route('/record_emotion', methods=['POST'])
def record_emotion():
    data = request.json
    emotion = data.get('emotion')
    if emotion in ['positive', 'negative']:
        if start_collection(emotion):
            return jsonify({'status': 'ok'})
        return jsonify({'status': 'error', 'message': 'Already recording or too soon'})
    return jsonify({'status': 'error', 'message': 'Invalid emotion'})

@app.route('/retrain', methods=['POST'])
def retrain():
    state.mode = 'training'
    state.model.force_retrain()
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌿 PHAENOMENA - EMOTIONSWAHRNEHMUNG")
    print("   Unified Exhibition Interface with Gesture Controls")
    print("="*60)
    print(f"🌐 http://localhost:5003")
    print("="*60 + "\n")
    
    if camera.start():
        state.camera_connected = True
        print("✅ Camera started")
    else:
        print("⚠️  Camera not available")
    
    loaded, age = state.model.load_latest()
    if loaded and age < MODEL_MAX_AGE_MINUTES:
        state.mode = 'exhibition'
        print(f"✨ Model fresh - exhibition mode")
    else:
        print("🔧 Training mode")
    
    threading.Thread(target=run_esp32_listener, daemon=True).start()
    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=prediction_loop, daemon=True).start()
    threading.Thread(target=broadcast_updates, daemon=True).start()
    
    socketio.run(app, host='0.0.0.0', port=5003, debug=False, allow_unsafe_werkzeug=True)
