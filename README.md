# 🌱 Plant Emotion Detector

**COIN Seminar 2026 – Team 1**

A biosensing + machine learning project exploring whether a plant's bioelectric
voltage signal carries information about a nearby human's emotional state.
A custom ESP32 sensor rig records plant voltage (plus ambient CO₂), while a
Polar H10 chest strap and a webcam capture the participant's heart-rate
variability and facial expression. These are combined into proxy emotion
labels, which are then used to train and compare ML models (Random Forest,
XGBoost, Neural Network) that predict emotion from the plant signal alone.

The full write-up, including methodology, results and limitations, is in
[project_report.pdf](project_report.pdf).

---

## Repository Structure

```
├── arduino/
│   └── plant_sensor_script.ino     ESP32 firmware: reads plant voltage (AD8232)
│                                    + CO2 (SCD4x), streams both over WebSocket
│
├── notebooks/                      Pipeline, run in numeric order
│   ├── 01_init_recording.ipynb     Records one session (Polar H10, camera, ESP32)
│   │                                → data/sessions/<session>/{events.jsonl,
│   │                                  session_meta.json, video.mp4}
│   ├── 02_emotion_labelling.ipynb  Turns one session's raw signals into per-window
│   │                                proxy emotion labels (HRV → arousal, FER → valence)
│   │                                → fer_events_mtcnn.jsonl, labelled_windows_ml.csv,
│   │                                  labelled_windows_overlapping.csv
│   ├── 03_dataset_creation.ipynb   Runs 02 over every session (jupyter nbconvert
│   │                                --execute) and merges the result
│   │                                → data/datasets/consolidated_dataset.csv
│   ├── 04_01/02/03_ml_pipeline_*.ipynb
│   │                                Train & compare RF / XGBoost / NN for emotion,
│   │                                arousal, and valence → models/*.pkl
│   └── 05_Correlation_Analysis.ipynb
│                                    Spearman correlation of the plant signal against
│                                    HRV, FER and CO2 (independent of the trained models)
│                                    → data/datasets/correlation_co2_valence_results.csv
│
├── models/                         Best model per axis (output of the 04_* notebooks)
│   ├── random_forest_emotion.pkl
│   ├── neural_network_arousal.pkl
│   └── random_forest_valence.pkl
│
├── data/
│   ├── sessions/                   15 recorded participant sessions (raw signals,
│   │                                per-session labels, executed 02 notebook copy)
│   ├── baseline/                   Plant-only reference recording (no participant)
│   └── datasets/                   Consolidated dataset + analysis result CSVs
│
├── phaenomena/
│   ├── emotion_live_dashboard.py   Flask app for live sensor visualisation and
│   │                                real-time inference with the trained models
│   └── assets/                     Static assets for the dashboard
│
├── project_report.pdf              Final seminar report
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
4. `05_Correlation_Analysis.ipynb` — correlation analysis between the plant signal
   and HRV / FER / CO₂

All 15 recorded sessions from the experiment are included in `data/datasets/consolidated_dataset.csv`, so step 4 
can be run directly without re-recording anything. Only the
raw session data is excluded.

To try the live dashboard (real-time sensor view + live model predictions):

```bash
python phaenomena/emotion_live_dashboard.py
```

then open `http://localhost:5004` in a browser. This needs the actual hardware connected —
set `ESP32_IP` near the top of the script to the address shown on the OLED display, and
make sure the Polar H10 and webcam are reachable.
