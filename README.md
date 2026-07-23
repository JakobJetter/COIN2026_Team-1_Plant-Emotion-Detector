# 🌱 Plant Emotion Detector

**COIN Seminar 2026 – Team 1**

A biosensing + machine learning project exploring whether a plant's bioelectric
voltage signal carries information about a nearby human's emotional state.
A custom ESP32 sensor rig records plant voltage (plus ambient CO₂), while a
Polar H10 chest strap and a webcam capture the participant's heart-rate
variability and facial expression. These are combined into proxy emotion
labels, which are then used to train and compare ML models (Random Forest,
XGBoost, Neural Network) that predict emotion from the plant signal alone.

---

## Repository Structure

```
├── arduino/
│   └── plant_sensor_script.ino     ESP32 firmware: reads plant voltage (AD8232)
│                                    + CO2 (SCD4x), streams both over WebSocket
│
├── notebooks/                      Pipeline, run in numeric order
│   ├── 01_init_recording.ipynb     Records one session (Polar H10, camera, ESP32)
│   │                                → data/sessions/<session>/{events.jsonl, video.mp4}
│   ├── 02_emotion_labelling.ipynb  Turns one session's raw signals into per-window
│   │                                proxy emotion labels (HRV → arousal, FER → valence)
│   ├── 03_dataset_creation.ipynb   Runs 02 over every session and merges the result
│   │                                → data/datasets/consolidated_dataset.csv
│   ├── 04_01/02/03_ml_pipeline_*.ipynb
│   │                                Train & compare RF / XGBoost / NN for emotion,
│   │                                arousal, and valence → models/*.pkl
│   └── 06_Correlation_Analysis_V02.ipynb
│                                    Spearman correlation of the plant signal against
│                                    HRV, FER and CO2 (independent of the trained models)
│
├── models/                         Trained model artifacts (output of the 04_* notebooks)
├── data/
│   ├── sessions/                   Raw per-session recordings
│   └── datasets/                   Consolidated dataset + analysis result CSVs
│
├── phaenomena/
│   └── emotion_live_dashboard.py   Flask app for live sensor visualisation and
│                                    real-time inference with the trained models
│
└── requirements.txt
```

---

## Hardware Setup

- **ESP32 DevKit** + **AD8232** bioelectric amplifier (plant leaf electrodes)
- **SCD4x** CO₂ sensor (I2C)
- **SSD1306** OLED display (status/IP)
- **Polar H10** chest strap (Bluetooth LE, HRV)
- Webcam for facial expression recognition (FER)
- Plant used: *Kalanchoe*

Flash `arduino/plant_sensor_script.ino` to the ESP32 (set your WiFi credentials at the
top of the file); the OLED display then shows the IP address needed in Notebook 01.

---

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then run the notebooks in order:

1. `01_init_recording.ipynb` — record a new session (skip if you just want to work with
   the already-recorded sessions in `data/sessions/`)
2. `03_dataset_creation.ipynb` — (re)generates labels for every session and builds
   `consolidated_dataset.csv` (calls `02_emotion_labelling.ipynb` internally, no need to
   run it manually)
3. `04_01_ml_pipeline_emotion.ipynb`, `04_02_ml_pipeline_arousal.ipynb`,
   `04_03_ml_pipeline_valence.ipynb` — train and compare the models
4. `05_Correlation_Analysis_V02.ipynb` — correlation analysis between the plant signal
   and HRV / FER / CO₂

`data/datasets/consolidated_dataset.csv` is already included, so steps 3 and 4 can be run
directly without re-recording or re-labelling anything.

To try the live dashboard (real-time sensor view + live model predictions):

```bash
python phaenomena/emotion_live_dashboard.py
```

then open `http://localhost:5004` in a browser.
