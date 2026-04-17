# 🌱 Plant Emotion Detector  
### COIN Seminar 2026 – Team 1

An interdisciplinary machine learning and biosensing project exploring how bioelectric signals can reflect human emotional states.

---

## 🧠 Project Idea

We investigate how plants can act as biological “sensors” of human emotions by measuring their electrical responses (voltage signals) during human interaction.

Humans emit multiple physiological signals linked to emotions, such as:
- Facial expressions
- Heart activity (HRV, EM field of the heart)
- Exhaled CO₂
- Volatile organic compounds (VOCs)

Plants respond to environmental stimuli with measurable bioelectric voltage changes.  
The central question is:

> Can we decode human emotions from plant voltage signals using machine learning (or improve existing algorithms)?

---

## 🎯 Objective

Build and compare machine learning models that predict or classify human emotional states using plant bioelectric time-series data.

We compare:

- Neural Networks (deep learning for time-series)
- XGBoost (gradient boosting)
- Random Forests (baseline ensemble model)

At the same time we extract our own data by measuring the voltage of the plant while measuring our emotions using not only the face impression. The core idea is to see if our data with more features create a higher accuracy in emotion detection then the available data with only face impressions.

---

## 📊 Available Data

The project is based on ~30 hours of exhibition data (“Silent Signals”), including:

- 🌿 Plant voltage time-series signals  
- 🎥 Video-based emotion annotations of participants ??? 
- 🌡️ Environmental and physiological signals (partially available / extended) ???

---

## 🧪 Scientific Motivation

Research suggests correlations between human stress and plant electrical activity (MDPI Biomimetics, 2025), but the mechanisms remain unclear.

We explore three possible channels:

- ⚡ Electromagnetic coupling (heart EM field influence)
- 🌬️ CO₂ mediation (breathing changes affecting plants)
- 🧪 VOC signaling (chemical emissions as key driver)

A large portion (~66%) of the interaction remains unexplained — an open research question.

---

## 🛠️ Technology Stack

- Python
- TensorFlow / PyTorch
- XGBoost
- Scikit-learn
- Time-series analysis tools
- ESP32 microcontroller
- AD8232 bioelectric sensor module
- VOC & CO₂ sensors
- Arduino IDE
- ...

---

## 🔧 Hardware System (Further explanations in the provoded document)

We built a low-cost plant biosensor system:

### Components:
- ESP32 DevKit
- AD8232 ECG module (plant bioelectric pickup)
- RC low-pass filter (~16 Hz cutoff)
- SSD1306 OLED display
- Gel ECG electrodes
- Breadboard setup

### Functionality:
- Measures plant voltage signals (µV range)
- Filters noise (50 Hz mains interference)
- Displays live signal data
- Connects electrode to plant leaf

Plants used:
- 🌿 Kalanchoe (primary)
- 🌸 Primrose
- 🌱 Tradescantia

---

## 🧪 System Pipeline

### Stream 1: Machine Learning
1. Data understanding  
2. Preprocessing & synchronization  
3. Feature engineering  
4. Model training (NN / XGBoost / RF)  
5. Evaluation & comparison  

### Stream 2: Field Experiments
1. Hardware assembly  
2. Signal testing  
3. Real-world data collection  
4. Integration with ML models  
5. Final evaluation  

---

## ⚠️ Challenges

- Noisy and inconsistent sensor data  
- Small dataset size  
- Hardware interference (EM noise, electrode contact issues)  
- High computational cost for training  
- Synchronization of multi-modal signals  

---

## 🔬 Expected Outcome

We aim to:

- Evaluate if plant bioelectric signals carry information about human emotions  
- Identify the best-performing ML model  
- Test whether additional physiological signals improve performance  
- Contribute to research in bio-inspired sensing and “honest signals”

--- 