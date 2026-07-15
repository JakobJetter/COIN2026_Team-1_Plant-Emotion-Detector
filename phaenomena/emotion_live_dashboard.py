#!/usr/bin/env python3
"""
Phaenomena - LIVE DASHBOARD (Multi-Sensor Overview)

Extends emotion_unified_phaenomena.py (kept untouched — used only as a design
and architecture reference) with a single dashboard that shows every live
data source at once:

  - Plant bioelectric voltage  (ESP32 / AD8232, WebSocket)
  - CO2 (ppm)                  (ESP32 / SCD4x, WebSocket)
  - Polar H10 heart rate       (Bluetooth LE)
  - Camera + HSEmotion         (facial emotion + valence/arousal recognition)
  - Plant-model predictions    (three pretrained ML models — emotion, arousal,
                                valence — loaded from ../models/, see
                                PlantAxisModel below)
  - Labelling-algorithm labels (the proxy-label rule from
                                notebooks/02_emotion_labelling.ipynb, run live
                                on the same HRV/FER signals)

Every live signal (voltage, CO2, heart rate, camera feed) updates continuously
as before. Only the two *predictions* per axis — the plant model's guess and
the labelling algorithm's proxy label — are window-based: both recompute once
every WIN_S (30s) of data has been collected, exactly like the non-overlapping
"ML grid" windows notebooks/02_emotion_labelling.ipynb builds from a full
recorded session. There is no on-site training here — the plant models are
loaded once for pure inference, and the labelling algorithm's calibration
(median/MAD per §4.7 of that notebook) is recomputed after each completed
window, growing across the live session in place of the notebook's
session-wide batch calibration (a live session has no fixed end to batch
over).

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
import warnings
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

# Cosmetic only: the persisted NeuralNet bundle is a Pipeline([scaler, mlp]);
# StandardScaler.transform() returns a plain ndarray, so the MLP step (fit on
# a named DataFrame) warns about losing feature names even though the column
# order is preserved end to end.
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

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

# HRV artefact correction (Lipponen & Tarvainen "Kubios" method), same
# library notebooks/02_emotion_labelling.ipynb §4.3 uses. Required, not
# optional, for this script (see requirements.txt).
import neurokit2 as nk

# ===== Face + emotion/valence/arousal detection (graceful cascade, mirrors
#       the original script's try/except chain but also works without
#       torch/facenet-pytorch by falling back to OpenCV's built-in Haar
#       cascade) =====
USE_HSEMOTION = False
HSEMOTION_MODEL = None
MTCNN_DETECTOR = None
HAAR_CASCADE = None
FACE_DETECTOR_MODE = None

# Same 8 expression classes and model family as notebooks/02_emotion_labelling.ipynb
# §4.6 (Savchenko, 2022) — 'enet_b0_8_va_mtl' predicts these 8 logits *plus*
# continuous valence/arousal in one forward pass, which the old discrete-only
# 'enet_b0_8_best_afew' model used here previously could not provide.
EMOTIONS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)


if CV2_OK:
    try:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer as HSEmotionONNX
        HSEMOTION_MODEL = HSEmotionONNX(model_name='enet_b0_8_va_mtl')
        _test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, _test_scores = HSEMOTION_MODEL.predict_emotions(_test_img, logits=True)
        USE_HSEMOTION = True
        print(f"  HSEmotion-ONNX ready (enet_b0_8_va_mtl, {np.asarray(_test_scores).size} outputs)")
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

# Firmware nominal (see arduino/plant_sensor_script.ino): 380Hz internal,
# decimated by 3 -> ~126.3Hz true rate. Matches
# notebooks/02_emotion_labelling.ipynb's PLANT_FS_NOMINAL_HZ. Each window
# still measures its own actual rate from arrival timestamps (see
# process_window below) rather than trusting this blindly.
PLANT_FS_NOMINAL_HZ = 380.0 / 3.01
ADC_VREF = 3.3
MAINS_HZ = 50.0                      # 46-54Hz bandstop before time-domain stats (§4.10)
PLANT_DEAD_VOLTAGE_THRESHOLD = 0.005  # volts

# ── Window geometry — mirrors notebooks/02_emotion_labelling.ipynb's
#    non-overlapping "ML grid" (STEP_S_EXPORT == WIN_S): every 30s of live
#    data becomes exactly one window, both for the plant models' input and
#    for the labelling algorithm's HRV/FER features.
WIN_S = 30
PLANT_BUFFER_SECONDS = 90            # kept in memory; comfortably covers one window + margin
PLANT_BUFFER_MAXLEN = int(PLANT_FS_NOMINAL_HZ * PLANT_BUFFER_SECONDS)
RR_BUFFER_MAXLEN = 600               # ~beats; comfortably covers one window at any real HR
FER_BUFFER_MAXLEN = 40 * 60          # ~60s of samples even at a fast camera loop

CO2_HISTORY_MAXLEN = 180
HR_HISTORY_MAXLEN = 60
CHART_VOLTAGE_POINTS = 400

UPDATE_RATE_MS = 150

# ── Calibration (§4.7 of the labelling notebook) — pragmatic, same values ──
K_TAU = 0.5
AROUSAL_K_TAU = 0.5
MIN_ML_WINDOWS = 3       # median/MAD unreliable with fewer completed windows

# ── Quality gates (same thresholds as the labelling notebook) ──────────────
HRV_ARTEFACT_FRACTION_MAX = 0.05
HRV_MIN_BEATS = 10
FER_COVERAGE_MIN = 0.5

# ── Label-confidence weights (§4.8), sum to 1 ───────────────────────────────
W1_DIRECTION_CLARITY = 0.4
W2_FER_CONFIDENCE = 0.3
W3_FACE_COVERAGE = 0.2
W4_AROUSAL_CLARITY = 0.1

# Three pretrained plant-only models (emotion / arousal / valence), each a
# {"model", "classes", "label_type", "model_type"} bundle produced by
# notebooks/04_01_ml_pipeline_emotion.ipynb / 04_02_ml_pipeline_arousal.ipynb /
# 04_03_ml_pipeline_valence.ipynb's "Persist the Best Model" section.
# Auto-discovered by filename suffix (*_emotion.pkl, *_arousal.pkl,
# *_valence.pkl) so whichever model type happened to win that training run
# is picked up without touching this file.
MODELS_DIR = Path(__file__).parent.parent / "models"

HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


# ═══════════════════════════════════════════════════════════════
# PLANT FEATURE EXTRACTION
# (ported verbatim from notebooks/02_emotion_labelling.ipynb §4.10, so the
#  live features are computed exactly like the ones each plant model in
#  ../models/ was trained on: mains-notched time-domain stats, raw-signal
#  saturation diagnostics, and the same four spectral bands + band_power_sum)
# ═══════════════════════════════════════════════════════════════

_trapz = getattr(np, "trapezoid", None) or np.trapz


def extract_plant_features(volts, fs):
    """Notebook-identical plant-voltage features for one WIN_S window."""
    out = {}
    v = np.asarray(volts, dtype=float)
    n = v.size
    out["plant_n_samples"] = int(n)

    if n == 0:
        for k in ["plant_mean", "plant_median", "plant_std", "plant_iqr", "plant_min", "plant_max",
                  "plant_range", "plant_rms", "plant_zcr", "plant_sat_pos_frac", "plant_sat_neg_frac",
                  "bp_0_1", "bp_1_5", "bp_5_15", "bp_15_30", "band_power_sum"]:
            out[k] = np.nan
        out["plant_const_flag"] = False
        out["plant_quality"] = "dead_flat"
        out["plant_dead_flag"] = True
        out["plant_suspect_flag"] = False
        return out

    # Mains notch (50Hz), time-domain stats only - saturation diagnostics and
    # the spectral computation below both deliberately stay on the raw signal.
    if n >= 32 and fs and fs > 2 * MAINS_HZ + 4:
        sos = sp_signal.butter(4, [MAINS_HZ - 4, MAINS_HZ + 4], btype="bandstop", fs=fs, output="sos")
        v_td = sp_signal.sosfiltfilt(sos, v)
    else:
        v_td = v

    mean = float(np.mean(v_td)); median = float(np.median(v_td))
    std = float(np.std(v_td, ddof=1)) if n > 1 else 0.0
    q75, q25 = np.percentile(v_td, [75, 25]); iqr = float(q75 - q25)
    vmin, vmax = float(np.min(v_td)), float(np.max(v_td)); rng_ = float(vmax - vmin)
    rms = float(np.sqrt(np.mean(v_td ** 2)))
    vd_td = v_td - mean
    zcr = float(np.mean(np.abs(np.diff(np.sign(vd_td))) > 0)) if n > 1 else 0.0
    out.update(plant_mean=round(mean, 6), plant_median=round(median, 6), plant_std=round(std, 6),
               plant_iqr=round(iqr, 6), plant_min=round(vmin, 6), plant_max=round(vmax, 6),
               plant_range=round(rng_, 6), plant_rms=round(rms, 6), plant_zcr=round(zcr, 6))

    sat_pos = float(np.mean(v >= ADC_VREF - 1e-3)); sat_neg = float(np.mean(v <= 1e-3))
    out.update(plant_sat_pos_frac=round(sat_pos, 4), plant_sat_neg_frac=round(sat_neg, 4),
               plant_const_flag=bool(np.all(v == v[0])))

    vd = v - float(np.mean(v))
    if n >= 16 and float(np.std(v)) > 0:
        nper = min(256, n)
        f, pxx = sp_signal.welch(vd, fs=fs, window="hann", nperseg=nper, scaling="density")

        def band(lo, hi):
            m = (f >= lo) & (f < hi)
            return float(_trapz(pxx[m], f[m])) if m.any() else 0.0

        bp_0_1, bp_1_5, bp_5_15, bp_15_30 = band(0, 1), band(1, 5), band(5, 15), band(15, 30)
        band_power_sum = bp_0_1 + bp_1_5 + bp_5_15 + bp_15_30
    else:
        bp_0_1 = bp_1_5 = bp_5_15 = bp_15_30 = band_power_sum = np.nan
    out.update(bp_0_1=bp_0_1, bp_1_5=bp_1_5, bp_5_15=bp_5_15, bp_15_30=bp_15_30,
               band_power_sum=band_power_sum)

    lsb = ADC_VREF / 4096.0
    std_counts = std / lsb if lsb else float("inf")
    if sat_pos > 0.95 or sat_neg > 0.95:
        q = "dead_saturated"
    elif std < PLANT_DEAD_VOLTAGE_THRESHOLD:
        q = "dead_flat"
    elif std_counts < 5:
        q = "dead_flat"
    elif (sat_pos + sat_neg) > 0.20:
        q = "suspect_clipped"
    elif std_counts < 10:
        q = "suspect_lowamp"
    else:
        q = "ok"
    out["plant_quality"] = q
    out["plant_dead_flag"] = q.startswith("dead")
    out["plant_suspect_flag"] = q.startswith("suspect")
    return out


def _feature_row_value(features, name):
    """Look up one model input column from a features dict, incl. the
    one-hot plant_quality dummies (e.g. 'quality_ok') a training run may
    have produced — derived from plant_quality itself, not stored directly."""
    if name.startswith("quality_"):
        return 1.0 if features.get("plant_quality") == name[len("quality_"):] else 0.0
    val = features.get(name)
    if val is None:
        return 0.0
    val = float(val)
    return 0.0 if val != val else val  # NaN check


# ═══════════════════════════════════════════════════════════════
# HRV FEATURES (ported from notebooks/02_emotion_labelling.ipynb §4.3/§4.5)
# ═══════════════════════════════════════════════════════════════

def correct_rr_series(rr_ms):
    """Lipponen & Tarvainen (2019) 'Kubios' beat correction via NeuroKit2.
    Returns (corrected_rr_ms, artefact_fraction)."""
    rr_ms = np.asarray(rr_ms, dtype=float)
    n = rr_ms.size
    if n < 4:
        return rr_ms, 0.0
    peaks = nk.intervals_to_peaks(rr_ms, sampling_rate=1000)
    try:
        info, peaks_clean = nk.signal_fixpeaks(peaks, sampling_rate=1000, method="Kubios")
    except Exception:
        return rr_ms, 0.0
    flagged_idx = set()
    for k in ("ectopic", "missed", "extra", "longshort"):
        flagged_idx.update(np.asarray(info.get(k, [])).astype(int).tolist())
    frac = len(flagged_idx) / max(1, n)
    rr_corrected = np.diff(np.asarray(peaks_clean, dtype=float))  # ms, sampling_rate=1000
    return rr_corrected, frac


def compute_hrv_features_window(rr_values_ms):
    n_raw = len(rr_values_ms)
    if n_raw < HRV_MIN_BEATS:
        return dict(rmssd=np.nan, mean_hr=np.nan, n_beats=n_raw, hrv_quality="poor")
    rr_corr, frac = correct_rr_series(rr_values_ms)
    if rr_corr.size < 2:
        return dict(rmssd=np.nan, mean_hr=np.nan, n_beats=int(rr_corr.size), hrv_quality="poor")
    diff = np.diff(rr_corr)
    rmssd = float(np.sqrt(np.mean(diff ** 2)))
    mean_hr = float(60_000.0 / np.mean(rr_corr))
    quality = "poor" if frac > HRV_ARTEFACT_FRACTION_MAX else "ok"
    return dict(rmssd=rmssd, mean_hr=mean_hr, n_beats=int(rr_corr.size), hrv_quality=quality)


# ═══════════════════════════════════════════════════════════════
# FER FEATURES (ported from notebooks/02_emotion_labelling.ipynb §4.6)
# ═══════════════════════════════════════════════════════════════

def compute_fer_features_window(fer_samples):
    n_sampled = len(fer_samples)
    with_face = [r for r in fer_samples if r["valence"] is not None]
    coverage = (len(with_face) / n_sampled) if n_sampled > 0 else 0.0
    if with_face:
        wts = np.array([r["confidence"] for r in with_face], dtype=float)
        wsum = wts.sum()
        wn = wts / wsum if wsum > 0 else np.ones_like(wts) / len(wts)
        val = float(np.clip((np.array([r["valence"] for r in with_face]) * wn).sum(), -1, 1))
        conf_mean = float(wts.mean())
        dom = Counter(r["emotion"] for r in with_face).most_common(1)[0][0]
    else:
        val = np.nan
        conf_mean = 0.0
        dom = None
    quality = "poor" if coverage < FER_COVERAGE_MIN else "ok"
    return dict(fer_valence=val, fer_confidence=conf_mean, fer_dominant_emotion=dom,
                face_coverage=round(coverage, 4), fer_n=n_sampled, fer_quality=quality)


# ═══════════════════════════════════════════════════════════════
# LIVE CALIBRATION + LABEL RULE
# (ported from notebooks/02_emotion_labelling.ipynb §4.7/§4.8. The notebook
#  calibrates once, in a batch, from a session's full set of ML-grid windows.
#  A live dashboard has no fixed session end to batch over, so this grows the
#  same median/MAD calibration window-by-window instead — the online analogue
#  of "session-wide", converging to the same thing by the time a session ends.)
# ═══════════════════════════════════════════════════════════════

class LiveCalibration:
    def __init__(self, k_tau=K_TAU, k_tau_arousal=AROUSAL_K_TAU, min_windows=MIN_ML_WINDOWS):
        self.k_tau = k_tau
        self.k_tau_arousal = k_tau_arousal
        self.min_windows = min_windows
        self.rmssd_hist = []
        self.hr_hist = []
        self.fer_valence_hist = []
        self.n_windows = 0

    def add_window(self, hrv, fer):
        self.n_windows += 1
        if hrv["hrv_quality"] != "poor" and not np.isnan(hrv["rmssd"]):
            self.rmssd_hist.append(hrv["rmssd"])
            self.hr_hist.append(hrv["mean_hr"])
        if fer["fer_quality"] != "poor" and not np.isnan(fer["fer_valence"]):
            self.fer_valence_hist.append(fer["fer_valence"])

    @staticmethod
    def _median_mad(values):
        if not values:
            return 0.0, 1e-9
        arr = np.asarray(values, dtype=float)
        med = float(np.median(arr))
        mad = max(1e-9, 1.4826 * float(np.median(np.abs(arr - med))))
        return med, mad

    def current(self):
        if self.n_windows < self.min_windows or len(self.rmssd_hist) < 2:
            return None
        med_rmssd, mad_rmssd = self._median_mad(self.rmssd_hist)
        med_hr, mad_hr = self._median_mad(self.hr_hist)
        med_val, mad_val = self._median_mad(self.fer_valence_hist)
        tau = self.k_tau * mad_val

        z_rmssd = [(r - med_rmssd) / mad_rmssd for r in self.rmssd_hist]
        z_hr = [(h - med_hr) / mad_hr for h in self.hr_hist]
        arousal_vals = [(-zr + zh) / 2 for zr, zh in zip(z_rmssd, z_hr)]
        med_arousal, mad_arousal = self._median_mad(arousal_vals)
        tau_arousal = self.k_tau_arousal * mad_arousal

        return dict(median_rmssd=med_rmssd, mad_rmssd=mad_rmssd, median_hr=med_hr, mad_hr=mad_hr,
                    tau=tau, median_fer_valence=med_val,
                    median_arousal_hrv=med_arousal, tau_arousal=tau_arousal)


def _direction_from_face(fer_valence, tau, median_valence):
    if fer_valence > median_valence + tau:
        return "positive"
    if fer_valence < median_valence - tau:
        return "negative"
    return "neutral"


def _level_from_hrv(arousal_hrv, tau_arousal, median_arousal):
    if arousal_hrv > median_arousal + tau_arousal:
        return "activated"
    if arousal_hrv < median_arousal - tau_arousal:
        return "calm"
    return "neutral"


def label_one_window(hrv, fer, calib):
    """Same rule as notebooks/02_emotion_labelling.ipynb §4.8, applied to a
    single live window instead of a whole dataframe."""
    if hrv["hrv_quality"] == "poor" or fer["fer_quality"] == "poor":
        return dict(status="excluded")
    if np.isnan(hrv["rmssd"]) or np.isnan(fer["fer_valence"]):
        return dict(status="excluded")

    rmssd_z = (hrv["rmssd"] - calib["median_rmssd"]) / calib["mad_rmssd"]
    hr_z = (hrv["mean_hr"] - calib["median_hr"]) / calib["mad_hr"]
    arousal_hrv = (-rmssd_z + hr_z) / 2

    valence_direction = _direction_from_face(fer["fer_valence"], calib["tau"], calib["median_fer_valence"])
    arousal_level = _level_from_hrv(arousal_hrv, calib["tau_arousal"], calib["median_arousal_hrv"])

    if valence_direction == "neutral" or arousal_level == "neutral":
        label = "neutral"
    else:
        label = f"{valence_direction} {arousal_level}"

    direction_clarity = (min(1.0, abs(fer["fer_valence"] - calib["median_fer_valence"]) / calib["tau"])
                          if calib["tau"] > 0 else 0.0)
    arousal_clarity = (min(1.0, abs(arousal_hrv - calib["median_arousal_hrv"]) / calib["tau_arousal"])
                        if calib["tau_arousal"] > 0 else 0.0)
    confidence = (W1_DIRECTION_CLARITY * direction_clarity + W2_FER_CONFIDENCE * fer["fer_confidence"]
                  + W3_FACE_COVERAGE * fer["face_coverage"] + W4_AROUSAL_CLARITY * arousal_clarity)

    return dict(status="ok", label=label, valence_direction=valence_direction,
                arousal_level=arousal_level, confidence=round(float(confidence), 3))


def _axis_view(algo_result, label_key):
    """Slice the combined labelling-algorithm result down to one axis."""
    if algo_result.get("status") != "ok":
        return algo_result
    return dict(status="ok", label=algo_result[label_key], confidence=algo_result["confidence"])


# ═══════════════════════════════════════════════════════════════
# PLANT MODELS — inference only, never retrained live. One per axis
# (emotion / arousal / valence), auto-discovered from ../models/.
# ═══════════════════════════════════════════════════════════════

class PlantAxisModel:
    """Loads whichever <model_type>_<label_type>.pkl exists in MODELS_DIR.

    Each pickle is a {"model", "classes", "label_type", "model_type"} bundle
    written by the "Persist the Best Model" section of
    notebooks/04_01_ml_pipeline_emotion.ipynb / 04_02_ml_pipeline_arousal.ipynb /
    04_03_ml_pipeline_valence.ipynb. If none exists yet for this axis (e.g. that
    notebook hasn't been run), predict() reports "unavailable" rather than
    falling back to a mock — there is no synthetic stand-in anymore.
    """

    def __init__(self, label_type):
        self.label_type = label_type
        self.model = None
        self.classes = None
        self.model_type = None
        self._load()

    def _load(self):
        if not MODELS_DIR.exists():
            print(f"Models directory not found: {MODELS_DIR}")
            return
        matches = sorted(MODELS_DIR.glob(f"*_{self.label_type}.pkl"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            print(f"No trained model found for '{self.label_type}' "
                  f"(looked for {MODELS_DIR}/*_{self.label_type}.pkl)")
            return
        path = matches[0]
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            self.model = bundle["model"]
            self.classes = bundle["classes"]
            self.model_type = bundle.get("model_type", "Model")
            print(f"Loaded {self.label_type} model: {self.model_type} ({path.name})")
        except Exception as e:
            print(f"Could not load {path}: {e}")

    @property
    def available(self):
        return self.model is not None

    def _feature_names(self):
        if hasattr(self.model, "feature_names_in_"):
            return list(self.model.feature_names_in_)
        if hasattr(self.model, "named_steps"):  # Pipeline([("scaler",...), ("model",...)])
            first = next(iter(self.model.named_steps.values()))
            if hasattr(first, "feature_names_in_"):
                return list(first.feature_names_in_)
        return None

    def predict(self, features):
        if not self.available:
            return dict(status="unavailable")
        names = self._feature_names()
        if not names:
            return dict(status="unavailable")
        row = pd.DataFrame([{n: _feature_row_value(features, n) for n in names}])
        try:
            probs = self.model.predict_proba(row)[0]
            idx = int(np.argmax(probs))
            prob_map = {self.classes[i]: float(p) for i, p in enumerate(probs)}
            return dict(status="ok", label=self.classes[idx], confidence=float(probs[idx]),
                        probs=prob_map, model_type=self.model_type)
        except Exception as e:
            print(f"{self.label_type} model prediction error: {e}")
            return dict(status="error")


# ═══════════════════════════════════════════════════════════════
# CAMERA + HSEMOTION (valence/arousal + 8-class expression)
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

        emotion, confidence, box, valence, arousal = (None, 0.0, None, None, None)
        if USE_HSEMOTION:
            emotion, confidence, box, valence, arousal = self._detect_emotion(frame)

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

        # Feed the window aggregator every processed frame - a row with
        # valence=None (no face) still counts toward face_coverage, exactly
        # like notebooks/02_emotion_labelling.ipynb's run_video_fer().
        ts_ms = time.time() * 1000
        if box is not None:
            state.fer_buffer.append({"ts": ts_ms, "emotion": emotion, "valence": valence,
                                      "arousal": arousal, "confidence": confidence})
        else:
            state.fer_buffer.append({"ts": ts_ms, "emotion": None, "valence": None,
                                      "arousal": None, "confidence": 0.0})

    def _detect_emotion(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box = self._find_face(frame, rgb)
            if box is None:
                return None, 0.0, None, None, None
            x1, y1, x2, y2 = box
            face_img = rgb[y1:y2, x1:x2]
            if face_img.size == 0:
                return None, 0.0, None, None, None
            # logits=True: raw outputs. We softmax the 8 expression logits
            # ourselves and read valence/arousal (indices 8/9) directly,
            # mirroring notebooks/02_emotion_labelling.ipynb §4.6 exactly -
            # the library's own argmax over all 10 raw values would be
            # meaningless here (it would include the regression outputs).
            _, scores = HSEMOTION_MODEL.predict_emotions(face_img, logits=True)
            scores = np.asarray(scores, dtype=np.float32).flatten()
            if scores.size >= 10:
                emo_logits = scores[:8]
                valence = float(np.clip(scores[8], -1.0, 1.0))
                arousal = float(np.clip(scores[9], -1.0, 1.0))
            else:
                emo_logits = scores[:8] if scores.size >= 8 else scores
                valence = arousal = None
            probs8 = _softmax(emo_logits)
            emo_idx = int(np.argmax(probs8))
            confidence = float(probs8.max())
            emotion = EMOTIONS[emo_idx] if emo_idx < len(EMOTIONS) else None
            return emotion, confidence, box, valence, arousal
        except Exception:
            return None, 0.0, None, None, None

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
        self.voltage_buffer = deque(maxlen=PLANT_BUFFER_MAXLEN)   # (ts_ms, volts)
        self.plant_quality = None

        self.latest_co2 = None
        self.co2_history = deque(maxlen=CO2_HISTORY_MAXLEN)

        self.polar_connected = False
        self.latest_bpm = None
        self.hr_history = deque(maxlen=HR_HISTORY_MAXLEN)
        self.rr_buffer = deque(maxlen=RR_BUFFER_MAXLEN)            # (ts_ms, rr_ms)

        self.fer_buffer = deque(maxlen=FER_BUFFER_MAXLEN)          # dicts, see CameraHandler

        # Window-based predictions - recomputed once per WIN_S, see process_window()
        self.calibration = LiveCalibration()
        self.session_t0_ms = None
        self.next_window_boundary_ms = None
        self.window_count = 0
        self.predictions = {
            "emotion": {"model": {"status": "unavailable"}, "algorithm": {"status": "calibrating",
                        "windows_collected": 0, "windows_needed": MIN_ML_WINDOWS}},
            "arousal": {"model": {"status": "unavailable"}, "algorithm": {"status": "calibrating",
                        "windows_collected": 0, "windows_needed": MIN_ML_WINDOWS}},
            "valence": {"model": {"status": "unavailable"}, "algorithm": {"status": "calibrating",
                        "windows_collected": 0, "windows_needed": MIN_ML_WINDOWS}},
        }


state = AppState()
camera = CameraHandler()
axis_models = {axis: PlantAxisModel(axis) for axis in ("emotion", "arousal", "valence")}


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
                    ts_ms = time.time() * 1000
                    for mv in data.get("voltages", []):
                        state.voltage_buffer.append((ts_ms, mv / 1000.0))
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
    ts_ms = time.time() * 1000
    while offset + 1 < len(data):
        rr_raw = struct.unpack_from("<H", data, offset)[0]
        rr_ms = rr_raw / 1024 * 1000
        if rr_ms > 0:
            bpm = 60000.0 / rr_ms
            state.latest_bpm = round(bpm, 1)
            state.hr_history.append(state.latest_bpm)
            state.rr_buffer.append((ts_ms, rr_ms))
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


def process_window(w_start, w_end):
    """Runs once per completed WIN_S window: plant features -> 3 plant-model
    predictions, and HRV+FER features -> one labelling-algorithm label,
    sliced into the same 3 axes for direct comparison."""
    plant_slice = [(ts, v) for ts, v in state.voltage_buffer if w_start <= ts < w_end]
    if len(plant_slice) >= 2:
        span_s = (plant_slice[-1][0] - plant_slice[0][0]) / 1000.0
        measured_fs = (len(plant_slice) - 1) / span_s if span_s > 0 else PLANT_FS_NOMINAL_HZ
    else:
        measured_fs = PLANT_FS_NOMINAL_HZ
    plant_features = extract_plant_features([v for _, v in plant_slice], fs=measured_fs)
    state.plant_quality = plant_features.get("plant_quality")

    rr_vals = [rr for ts, rr in state.rr_buffer if w_start <= ts < w_end]
    hrv = compute_hrv_features_window(rr_vals)

    fer_samples = [r for r in state.fer_buffer if w_start <= r["ts"] < w_end]
    fer = compute_fer_features_window(fer_samples)

    state.calibration.add_window(hrv, fer)
    calib = state.calibration.current()
    if calib is None:
        algo_result = dict(status="calibrating", windows_collected=state.calibration.n_windows,
                            windows_needed=MIN_ML_WINDOWS)
    else:
        algo_result = label_one_window(hrv, fer, calib)

    model_results = {axis: axis_models[axis].predict(plant_features) for axis in axis_models}

    state.predictions = {
        "emotion": {"model": model_results["emotion"], "algorithm": _axis_view(algo_result, "label")},
        "arousal": {"model": model_results["arousal"], "algorithm": _axis_view(algo_result, "arousal_level")},
        "valence": {"model": model_results["valence"], "algorithm": _axis_view(algo_result, "valence_direction")},
    }
    state.window_count += 1


def window_loop():
    while True:
        time.sleep(0.5)
        now = time.time() * 1000
        if state.session_t0_ms is None:
            state.session_t0_ms = now
            state.next_window_boundary_ms = now + WIN_S * 1000
            continue
        while now >= state.next_window_boundary_ms:
            w_end = state.next_window_boundary_ms
            w_start = w_end - WIN_S * 1000
            try:
                process_window(w_start, w_end)
            except Exception as e:
                print(f"Window processing error: {e}")
            state.next_window_boundary_ms += WIN_S * 1000


def broadcast_updates():
    while True:
        try:
            with camera.lock:
                face_detected = camera.face_detected
                face_emotion = camera.current_emotion
                face_confidence = camera.current_confidence

            now = time.time() * 1000
            seconds_until_next = (max(0.0, (state.next_window_boundary_ms - now) / 1000)
                                   if state.next_window_boundary_ms else None)

            socketio.emit('update', {
                'plant_connected': state.plant_connected,
                'camera_available': camera.available,
                'polar_connected': state.polar_connected,

                'voltages': [round(v * 1000, 2) for _, v in list(state.voltage_buffer)[-CHART_VOLTAGE_POINTS:]],
                'plant_quality': state.plant_quality,

                'co2': state.latest_co2,
                'co2_history': list(state.co2_history),

                'bpm': state.latest_bpm,
                'hr_history': list(state.hr_history),

                'face_detected': face_detected,
                'face_emotion': face_emotion,
                'face_confidence': face_confidence,

                'predictions': state.predictions,
                'window_count': state.window_count,
                'window_duration_s': WIN_S,
                'seconds_until_next_window': seconds_until_next,
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
            display: grid; grid-template-columns: 1.05fr 1.5fr 0.9fr; grid-template-rows: minmax(0, 1fr);
            gap: 16px; flex: 1.25 1 0; min-height: 0;
        }
        .charts-grid {
            display: grid; grid-template-columns: 2fr 1fr; grid-template-rows: minmax(0, 1fr);
            gap: 16px; flex: 0.85 1 0; min-height: 0;
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
        .panel-title .panel-subtitle {
            font-size: 0.85em; font-weight: 500; letter-spacing: 0; text-transform: none; color: var(--text-dim);
        }

        /* Camera panel */
        .camera-container { width: 100%; flex: 1; min-height: 0; border-radius: 12px; overflow: hidden; background: #000; }
        .camera-container img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .no-camera { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-dim); font-size: 0.85em; }
        .face-readout { margin-top: 8px; text-align: center; flex-shrink: 0; }
        .face-emotion { font-size: 1.2em; font-weight: 700; }
        .face-confidence { margin-top: 2px; font-size: 0.8em; color: var(--text-dim); }

        /* Predictions panel: 3 axis rows, each = plant-model card vs algorithm card */
        .pred-rows { display: flex; flex-direction: column; justify-content: space-between; flex: 1; min-height: 0; gap: 6px; }
        .pred-axis-block { flex: 1; min-height: 0; display: flex; flex-direction: column; justify-content: center; }
        .pred-axis-name { font-size: 0.68em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim); margin-bottom: 4px; }
        .pred-row { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 8px;
                    background: rgba(0,0,0,0.18); border-radius: 10px; padding: 7px 12px; }
        .pred-card { text-align: center; min-width: 0; }
        .pred-card .source { font-size: 0.62em; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-dim); margin-bottom: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .pred-card .value { font-size: 1.02em; font-weight: 700; text-transform: capitalize; line-height: 1.2; }
        .pred-card .meta { font-size: 0.68em; color: var(--text-dim); margin-top: 1px; }
        .pred-model-badge { opacity: 0.75; font-weight: 600; }
        .pred-vs { font-size: 0.62em; color: var(--text-dim); opacity: 0.55; }
        .pred-status { margin-top: 8px; text-align: center; font-size: 0.72em; color: var(--text-dim); flex-shrink: 0; }

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
        .chart-stat .quality-tag { font-size: 0.72em; color: var(--text-dim); }

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

        <!-- Predictions: plant model vs labelling algorithm, per axis -->
        <div class="panel">
            <div class="panel-title">Predictions <span class="panel-subtitle" id="predHeaderSub">Plant Model vs Labelling Algorithm</span></div>
            <div class="pred-rows">
                <div class="pred-axis-block">
                    <div class="pred-axis-name">Emotion</div>
                    <div class="pred-row">
                        <div class="pred-card" id="predEmotionModel"></div>
                        <div class="pred-vs">vs</div>
                        <div class="pred-card" id="predEmotionAlgo"></div>
                    </div>
                </div>
                <div class="pred-axis-block">
                    <div class="pred-axis-name">Arousal</div>
                    <div class="pred-row">
                        <div class="pred-card" id="predArousalModel"></div>
                        <div class="pred-vs">vs</div>
                        <div class="pred-card" id="predArousalAlgo"></div>
                    </div>
                </div>
                <div class="pred-axis-block">
                    <div class="pred-axis-name">Valence</div>
                    <div class="pred-row">
                        <div class="pred-card" id="predValenceModel"></div>
                        <div class="pred-vs">vs</div>
                        <div class="pred-card" id="predValenceAlgo"></div>
                    </div>
                </div>
            </div>
            <div class="pred-status" id="predStatus">Collecting first window...</div>
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
            <div class="chart-stat">
                <span class="value plant" id="voltageValue">--<span class="unit">mV</span></span>
                <span class="quality-tag" id="plantQualityTag">&nbsp;</span>
            </div>
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

        // Same colour convention as notebooks/02_emotion_labelling.ipynb §4.13 and
        // notebooks/03_dataset_creation.ipynb - same colour, same meaning, everywhere.
        const EMOTION_COLORS = {
            'positive activated': '#2ecc71', 'positive calm': '#1e8449',
            'negative activated': '#e74c3c', 'negative calm': '#943126',
            'neutral': '#7f8c8d'
        };
        const VALENCE_COLORS = { positive: '#27ae60', neutral: '#7f8c8d', negative: '#c0392b' };
        const AROUSAL_COLORS = { activated: '#e67e22', neutral: '#7f8c8d', calm: '#2980b9' };
        const AXIS_COLORS = { emotion: EMOTION_COLORS, arousal: AROUSAL_COLORS, valence: VALENCE_COLORS };

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
        function titleCase(s) { return s ? s.replace(/(^|\\s)\\S/g, c => c.toUpperCase()) : s; }
        function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

        function renderPredCard(elId, axis, entry, sourceLabel) {
            const el = document.getElementById(elId);
            entry = entry || { status: 'unavailable' };
            if (entry.status === 'unavailable') {
                el.innerHTML = `<div class="source">${sourceLabel}</div><div class="value" style="color:var(--text-dim)">Not trained yet</div>`;
                return;
            }
            if (entry.status === 'calibrating') {
                el.innerHTML = `<div class="source">${sourceLabel}</div><div class="value" style="color:var(--text-dim)">Calibrating&hellip;</div><div class="meta">${entry.windows_collected}/${entry.windows_needed} windows</div>`;
                return;
            }
            if (entry.status === 'excluded' || entry.status === 'error') {
                el.innerHTML = `<div class="source">${sourceLabel}</div><div class="value" style="color:var(--text-dim)">Excluded</div>`;
                return;
            }
            const color = (AXIS_COLORS[axis] || {})[entry.label] || '#7f8c8d';
            const badge = entry.model_type ? ` <span class="pred-model-badge">(${entry.model_type})</span>` : '';
            el.innerHTML = `<div class="source">${sourceLabel}${badge}</div>` +
                            `<div class="value" style="color:${color}">${titleCase(entry.label)}</div>` +
                            `<div class="meta">${pct(entry.confidence)}</div>`;
        }

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
                document.getElementById('faceConfidence').textContent = ' ';
            }

            // Predictions: plant model vs labelling algorithm, per axis
            const preds = d.predictions || {};
            ['emotion', 'arousal', 'valence'].forEach(axis => {
                const p = preds[axis] || {};
                renderPredCard('pred' + capitalize(axis) + 'Model', axis, p.model, 'Plant Model');
                renderPredCard('pred' + capitalize(axis) + 'Algo', axis, p.algorithm, 'Labelling Algorithm');
            });
            const secs = d.seconds_until_next_window;
            document.getElementById('predStatus').textContent = d.window_count
                ? `Window #${d.window_count} complete · next update in ${secs != null ? Math.round(secs) : '--'}s`
                : (secs != null ? `Collecting first window · ${Math.round(secs)}s left` : 'Collecting first window...');

            // Heart rate
            document.getElementById('bpmValue').textContent = d.bpm != null ? d.bpm.toFixed(0) : '--';
            drawLine(document.getElementById('hrCanvas'), d.hr_history, '#ff6ac8');

            // Plant voltage
            document.getElementById('voltageValue').innerHTML = (d.voltages && d.voltages.length)
                ? d.voltages[d.voltages.length - 1].toFixed(1) + '<span class="unit">mV</span>' : '--';
            document.getElementById('plantQualityTag').textContent = d.plant_quality ? ('quality: ' + d.plant_quality) : '';
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
    threading.Thread(target=window_loop, daemon=True).start()
    threading.Thread(target=broadcast_updates, daemon=True).start()

    socketio.run(app, host='0.0.0.0', port=5004, debug=False, allow_unsafe_werkzeug=True)
