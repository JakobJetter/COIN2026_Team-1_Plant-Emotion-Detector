/*
 * ESP32 Plant Sensor - Streaming with Display (Filtered Version)
 * 
 * Features:
 * - Real-time voltage streaming via WebSocket
 * - Aggressive digital lowpass filtering (~8Hz cutoff to match Oxocard)
 * - OLED display showing IP and streaming status
 * - Non-blocking WiFi reconnect (router can start before or after sensor)
 * 
 * Hardware: ESP32 DevKit + AD8232 + SSD1306 OLED Display
 * 
 * AD8232 Connections:
 * - 3.3V → 3.3V rail
 * - GND → GND rail  
 * - Output → GPIO0 (A0) - adjust for your board
 * - SDN → 3.3V (keep active)
 * - Plant electrodes via 3.5mm jack
 * 
 * OLED Display Connections (I2C):
 * - SDA → GPIO8 (adjust for your board - GPIO21 on DevKit C)
 * - SCL → GPIO9 (adjust for your board - GPIO22 on DevKit C)
 * - VCC → 3.3V
 * - GND → GND
 */

#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <SensirionI2CSgp40.h>
#include <VOCGasIndexAlgorithm.h>
#include <SensirionI2cScd4x.h>

VOCGasIndexAlgorithm voc_algorithm;
SensirionI2CSgp40 sgp40;
SensirionI2cScd4x scd4x;

char errorMessage[256];

// ===== WIFI CONFIGURATION =====
const char* ssid =  "PixelDingens";
const char* password = "SoederIstEinSaftsack";

// ===== WEBSOCKET SERVER =====
WebSocketsServer webSocket = WebSocketsServer(81);  // WebSocket on port 81

// ===== OLED DISPLAY CONFIGURATION =====
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ===== SAMPLING CONFIGURATION =====
const int SAMPLE_RATE = 380;           // 380Hz internal sampling
const int OUTPUT_RATE = 100;           // 100Hz output rate (less decimation)
const int SAMPLE_INTERVAL_US = 1000000 / SAMPLE_RATE;
const int DECIMATION_FACTOR = SAMPLE_RATE / OUTPUT_RATE;  // ~4

// ===== ADC CONFIGURATION - AD8232 Optimized =====
const int ADC_PIN = 34;  // Adjust for your board. 34 for ESP32D, A0 for ESP32C
const int ADC_RESOLUTION = 12;
const float ADC_VOLTAGE_REF = 3.3;

// ===== STREAMING BUFFER =====
const int BUFFER_SIZE = 50;  // Send 50 samples at a time (500ms at 100Hz)
float voltageBuffer[BUFFER_SIZE];
int bufferIndex = 0;

// ===== TIMING =====
unsigned long lastSampleTime = 0;
unsigned long lastBatchSend = 0;
const int BATCH_SEND_INTERVAL_MS = 500;

// ===== WEBSOCKET CLIENT TRACKING =====
int connectedClients = 0;
int packetsSent = 0;

// ===== STATUS VARIABLES =====
unsigned long totalSamples = 0;
unsigned long outputSamples = 0;
float lastVoltage = 0.0;
float lastFilteredVoltage = 0.0;
float lastVOC = 0.0;
float lastCO2 = 0.0;
String wifi_status = "Disconnected";
String websocket_status = "Stopped";

// Gas sensor timing (non-blocking)
unsigned long lastGasMeasure = 0;
const unsigned long GAS_MEASURE_INTERVAL_MS = 1000;

// ===== DISPLAY UPDATE TIMING =====
unsigned long lastDisplayUpdate = 0;
const unsigned long DISPLAY_UPDATE_INTERVAL = 2000;

// ===== WIFI RECONNECT TIMING =====
unsigned long lastWifiCheck = 0;
const unsigned long WIFI_CHECK_INTERVAL = 5000;   // check every 5s
bool webSocketStarted = false;

// ═══════════════════════════════════════════════
// DIGITAL FILTER CONFIGURATION
// ═══════════════════════════════════════════════
// 
// Two-stage filtering to match Oxocard's 0.07-8.8Hz bandpass:
// 1. IIR Lowpass at ~8Hz (removes 50Hz noise and high-freq hash)
// 2. IIR Highpass at ~0.1Hz (removes DC drift)
//
// Filter coefficients calculated for 380Hz sample rate

// === STAGE 1: 20Hz Lowpass ===
const float LP_ALPHA = 0.25;  // Lowpass coefficient (~20Hz cutoff)
float lpFilterState = 0.0;

// === STAGE 2: 0.1Hz Highpass (DC blocking) ===
const float HP_ALPHA = 0.998;  // Highpass coefficient (~0.1Hz cutoff)
float hpFilterState = 0.0;
float hpPrevInput = 0.0;

// === Decimation counter ===
int decimationCounter = 0;
float decimationAccumulator = 0.0;

// ===== FUNCTION DECLARATIONS =====
void setupOLED();
void updateDisplay();
void setupADC();
void startWiFi();
void checkWiFiAndReconnect();
float readAndFilterSensor();
void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length);
void sendBatchData();
float adcToVoltage(int adcValue);
float applyLowpass(float input);
float applyHighpass(float input);
void i2cScan();

// ═══════════════════════════════════════════════
// SETUP
// ═══════════════════════════════════════════════

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n╔══════════════════════════════════════════════╗");
    Serial.println("║  ESP32 Plant Sensor - Filtered Streaming     ║");
    Serial.println("╚══════════════════════════════════════════════╝");
    Serial.println();
    Serial.println("Features:");
    Serial.println("  • Internal sampling: 380Hz");
    Serial.println("  • Output rate: 50Hz (filtered & decimated)");
    Serial.println("  • Bandpass filter: 0.1-8Hz (matches Oxocard)");
    Serial.println("  • AD8232 bioelectric amplifier");
    Serial.println("  • Non-blocking WiFi reconnect");
    Serial.println();
    
    setupOLED();
    setupADC();
    setupVOC();
    // Initialize SCD4x CO2 sensor
    // SCD4x initialized inside setupVOC to share I2C
    
    // Start WiFi non-blocking — does NOT wait, does NOT freeze
    startWiFi();
    
    // Initialize filter states with first reading
    int initialReading = analogRead(ADC_PIN);
    float initialVoltage = adcToVoltage(initialReading);
    lpFilterState = initialVoltage;
    hpPrevInput = initialVoltage;
    hpFilterState = 0.0;
    
    lastSampleTime = micros();
    lastBatchSend = millis();
    lastWifiCheck = millis();
    
    Serial.println("Sensor running — waiting for WiFi...\n");
}

// ═══════════════════════════════════════════════
// LOOP
// ═══════════════════════════════════════════════

void loop() {
    // ── WiFi watchdog: reconnect if lost ──────────────────
    if (millis() - lastWifiCheck >= WIFI_CHECK_INTERVAL) {
        lastWifiCheck = millis();
        checkWiFiAndReconnect();
    }

    // ── WebSocket: start once WiFi is up, keep running ────
    if (WiFi.status() == WL_CONNECTED) {
        if (!webSocketStarted) {
            webSocket.begin();
            webSocket.onEvent(webSocketEvent);
            webSocketStarted = true;
            websocket_status = "Ready";
            Serial.printf("✓ WebSocket server started on ws://%s:81\n",
                          WiFi.localIP().toString().c_str());
        }
        webSocket.loop();
    }

    uint16_t error;
    uint16_t srawVoc = 0;
    uint16_t defaultCompensationRh = 0x8000;  // in ticks as defined by SGP40
    uint16_t defaultCompensationT = 0x6666;   // in ticks as defined by SGP40

    // 1. Sleep: Measure every second (1Hz), as defined by the Gas Index
    // Algorithm
    //    prerequisite
    // Non-blocking gas sensor reads (SGP40 + SCD4x) at ~1Hz
    unsigned long nowMs = millis();
    if (nowMs - lastGasMeasure >= GAS_MEASURE_INTERVAL_MS) {
        lastGasMeasure = nowMs;

        error = sgp40.measureRawSignal(defaultCompensationRh, defaultCompensationT, srawVoc);
        if (error) {
            Serial.print("SGP40 - Error trying to execute measureRawSignals(): ");
            errorToString(error, errorMessage, 256);
            Serial.println(errorMessage);
            // Run I2C scan to help diagnose connectivity/address issues
            i2cScan();
        } else {
            int32_t vocIndex = voc_algorithm.process(srawVoc);
            Serial.print("VOC Index: ");
            Serial.print(vocIndex);
            Serial.print("\n");
            lastVOC = vocIndex;
        }

        // Read SCD4x CO2 measurement (poll data-ready, then read)
        bool dataReady = false;
        int scdErr = scd4x.getDataReadyStatus(dataReady);
        if (scdErr != 0) {
            Serial.printf("SCD4x getDataReadyStatus error: %d (0x%X)\n", scdErr, scdErr);
            // Run I2C scan to help diagnose connectivity/address issues
            i2cScan();
        } else if (dataReady) {
            uint16_t co2ppm = 0;
            float scdTemp = 0.0;
            float scdRh = 0.0;
            scdErr = scd4x.readMeasurement(co2ppm, scdTemp, scdRh);
            if (scdErr == 0) {
                lastCO2 = co2ppm;
                Serial.print("CO2 (ppm): ");
                Serial.println(lastCO2);
            } else {
                Serial.print("SCD4x read error: ");
                Serial.println(scdErr);
            }
        }
    }

    // ── Sensor sampling at 380Hz (always, regardless of WiFi) ──
    unsigned long currentTime = micros();
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_US) {
        lastSampleTime = currentTime;
        totalSamples++;
        
        int rawADC = analogRead(ADC_PIN);
        float rawVoltage = adcToVoltage(rawADC);
        lastVoltage = rawVoltage;
        
        float lpFiltered = applyLowpass(rawVoltage);
        float filtered = applyHighpass(lpFiltered);
        (void)filtered;  // highpass computed but output uses lpFiltered (DC visible)
        
        float outputVoltage = lpFiltered;
        lastFilteredVoltage = outputVoltage;
        
        decimationAccumulator += outputVoltage;
        decimationCounter++;
        
        if (decimationCounter >= DECIMATION_FACTOR) {
            float decimatedVoltage = decimationAccumulator / DECIMATION_FACTOR;
            outputSamples++;
            
            voltageBuffer[bufferIndex] = decimatedVoltage;
            bufferIndex++;
            
            decimationAccumulator = 0.0;
            decimationCounter = 0;
            
            if (bufferIndex >= BUFFER_SIZE) {
                sendBatchData();
                bufferIndex = 0;
            }
        }
        
        if (totalSamples % 1000 == 0) {
            Serial.printf("[Sample %lu] Raw: %.1f mV | Filtered: %.1f mV | Out: %lu\n",
                          totalSamples, rawVoltage, lastFilteredVoltage, outputSamples);
        }
    }
    
    // ── Backup batch send ─────────────────────────────────
    if (bufferIndex > 0 && (millis() - lastBatchSend) >= BATCH_SEND_INTERVAL_MS) {
        sendBatchData();
        bufferIndex = 0;
    }
    
    // ── Display update ────────────────────────────────────
    if (millis() - lastDisplayUpdate >= DISPLAY_UPDATE_INTERVAL) {
        updateDisplay();
        lastDisplayUpdate = millis();
    }
}

void setupVOC() {

    Serial.begin(115200);
    delay(200);

    // Start I2C on custom pins (use same pins as OLED)
    Wire.begin(21, 22);   // SDA = 21, SCL = 22

    sgp40.begin(Wire);
    // Initialize SCD4x CO2 sensor (use official driver API)
    scd4x.begin(Wire, 0x62);
    // Start periodic measurement on SCD4x (library will handle interval)
    scd4x.startPeriodicMeasurement();

    delay(1000);
    // Print I2C devices at startup to help verify sensor connections
    i2cScan();
    // If no devices found when using custom pins, retry using default I2C pins
    Serial.println("Retrying I2C scan with default Wire.begin() pins...");
    Wire.begin();
    i2cScan();

    int32_t index_offset;
    int32_t learning_time_offset_hours;
    int32_t learning_time_gain_hours;
    int32_t gating_max_duration_minutes;
    int32_t std_initial;
    int32_t gain_factor;
    voc_algorithm.get_tuning_parameters(
        index_offset, learning_time_offset_hours, learning_time_gain_hours,
        gating_max_duration_minutes, std_initial, gain_factor);

    Serial.println("\nVOC Gas Index Algorithm parameters");
    Serial.print("Index offset:\t");
    Serial.println(index_offset);
    Serial.print("Learning time offset hours:\t");
    Serial.println(learning_time_offset_hours);
    Serial.print("Learning time gain hours:\t");
    Serial.println(learning_time_gain_hours);
    Serial.print("Gating max duration minutes:\t");
    Serial.println(gating_max_duration_minutes);
    Serial.print("Std inital:\t");
    Serial.println(std_initial);
    Serial.print("Gain factor:\t");
    Serial.println(gain_factor);
}

// ═══════════════════════════════════════════════
// WIFI — NON-BLOCKING
// ═══════════════════════════════════════════════

void startWiFi() {
    Serial.printf("WiFi: connecting to '%s' (non-blocking)...\n", ssid);
    wifi_status = "Connecting";
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    // Returns immediately — connection happens in background
}

void checkWiFiAndReconnect() {
    if (WiFi.status() == WL_CONNECTED) {
        // All good — update status string if it was "Connecting"
        if (wifi_status != "Connected") {
            wifi_status = "Connected";
            Serial.printf("✓ WiFi connected — IP: %s  Signal: %d dBm\n",
                          WiFi.localIP().toString().c_str(), WiFi.RSSI());
        }
        return;
    }

    // Not connected — restart connection attempt
    wifi_status = "Reconnecting";
    webSocketStarted = false;   // WebSocket will re-init once WiFi is back
    websocket_status = "Waiting";
    Serial.printf("WiFi lost — reconnecting to '%s'...\n", ssid);
    WiFi.disconnect();
    delay(100);
    WiFi.begin(ssid, password);
    // Again non-blocking — result checked next interval
}

// ═══════════════════════════════════════════════
// DIGITAL FILTERS
// ═══════════════════════════════════════════════

float applyLowpass(float input) {
    lpFilterState = LP_ALPHA * input + (1.0 - LP_ALPHA) * lpFilterState;
    return lpFilterState;
}

float applyHighpass(float input) {
    float output = HP_ALPHA * (hpFilterState + input - hpPrevInput);
    hpFilterState = output;
    hpPrevInput = input;
    return output;
}

// Simple I2C scanner to print devices on the bus for diagnostics
void i2cScan() {
    Serial.println("Running I2C scan...");
    byte error, address;
    int nDevices = 0;

    for (address = 1; address < 127; address++) {
        Wire.beginTransmission(address);
        error = Wire.endTransmission();

        if (error == 0) {
            Serial.print("I2C device found at 0x");
            if (address < 16) Serial.print("0");
            Serial.print(address, HEX);
            Serial.println(" !");
            nDevices++;
        } else if (error == 4) {
            Serial.print("Unknown error at 0x");
            if (address < 16) Serial.print("0");
            Serial.println(address, HEX);
        }
    }

    if (nDevices == 0)
        Serial.println("No I2C devices found");
    else
        Serial.println("I2C scan complete");
}

// ═══════════════════════════════════════════════
// DISPLAY
// ═══════════════════════════════════════════════

void setupOLED() {
    Serial.print("Initializing OLED display... ");
    Wire.begin(21, 22);  // SDA=21, SCL=22 (adjust per board)
    
    if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
        Serial.println("✗ FAILED — continuing without display");
        return;
    }
    Serial.println("✓");
    
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("Plant Sensor");
    display.println("(Filtered)");
    display.println();
    display.println("Initializing...");
    display.display();
    delay(1000);
}

void updateDisplay() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    
    // Line 1: Title
    display.setCursor(0, 0);
    display.println("Plant (Filtered)");
    
    // Line 2: WiFi status / IP
    display.setCursor(0, 12);
    if (WiFi.status() == WL_CONNECTED) {
        display.print("IP:");
        display.println(WiFi.localIP().toString().c_str());
    } else {
        display.print("WiFi: ");
        display.println(wifi_status);
    }
    
    // Line 3: WebSocket clients
    display.setCursor(0, 24);
    display.print("Clients: ");
    display.println(connectedClients);

    // Line 4: Plant filtered voltage
    display.setCursor(0, 36);
    display.print("V: ");
    display.print(lastFilteredVoltage, 1);
    display.println(" mV");

    // Line 5: VOC index
    display.setCursor(0, 48);
    display.print("VOC: ");
    display.print(lastVOC, 1);

    // Line 6: CO2 (ppm)
    display.setCursor(64, 48);
    display.print("CO2: ");
    display.print(lastCO2, 0);
    display.display();
}

// ═══════════════════════════════════════════════
// ADC
// ═══════════════════════════════════════════════

void setupADC() {
    analogReadResolution(ADC_RESOLUTION);
    analogSetAttenuation(ADC_11db);
    Serial.println("✓ ADC configured for AD8232");
    Serial.printf("  Resolution: %d bits\n", ADC_RESOLUTION);
    Serial.printf("  Reference: %.1f V\n", ADC_VOLTAGE_REF);
}

float adcToVoltage(int adcValue) {
    return (adcValue / 4095.0) * ADC_VOLTAGE_REF * 1000.0;
}

// ═══════════════════════════════════════════════
// WEBSOCKET
// ═══════════════════════════════════════════════

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
    switch (type) {
        case WStype_DISCONNECTED:
            connectedClients = max(0, connectedClients - 1);
            websocket_status = String(connectedClients) + " client(s)";
            Serial.printf("WebSocket: Client #%u disconnected (total: %d)\n",
                          num, connectedClients);
            break;
            
        case WStype_CONNECTED: {
            IPAddress ip = webSocket.remoteIP(num);
            connectedClients++;
            websocket_status = String(connectedClients) + " client(s)";
            Serial.printf("WebSocket: Client #%u connected from %s (total: %d)\n",
                          num, ip.toString().c_str(), connectedClients);
            
            StaticJsonDocument<256> statusDoc;
            statusDoc["type"]         = "status";
            statusDoc["message"]      = "Connected to ESP32 Plant Sensor (Filtered)";
            statusDoc["sampleRate"]   = OUTPUT_RATE;
            statusDoc["internalRate"] = SAMPLE_RATE;
            statusDoc["batchSize"]    = BUFFER_SIZE;
            statusDoc["filterLow"]    = "0.1Hz";
            statusDoc["filterHigh"]   = "8Hz";
            statusDoc["voc"]          = lastVOC;
            statusDoc["co2"]          = lastCO2;
            
            String statusJson;
            serializeJson(statusDoc, statusJson);
            webSocket.sendTXT(num, statusJson);
            break;
        }
        
        case WStype_TEXT:
            Serial.printf("WebSocket: Received from #%u: %s\n", num, payload);
            break;

        default:
            break;
    }
}

void sendBatchData() {
    if (connectedClients == 0 || bufferIndex == 0) return;
    
    StaticJsonDocument<1024> doc;
    doc["type"]       = "data";
    doc["timestamp"]  = millis();
    doc["sampleRate"] = OUTPUT_RATE;
    doc["samples"]    = bufferIndex;
    doc["voc"]        = lastVOC;   // ← latest VOC index from SGP40
    doc["co2"]        = lastCO2;  // ← latest CO2 ppm from SCD4x
    
    JsonArray voltages = doc.createNestedArray("voltages");
    for (int i = 0; i < bufferIndex; i++) {
        voltages.add(voltageBuffer[i]);
    }
    
    String jsonString;
    serializeJson(doc, jsonString);
    webSocket.broadcastTXT(jsonString);
    packetsSent++;
    lastBatchSend = millis();
    
    if (packetsSent % 10 == 0) {
        Serial.printf("WebSocket: Packet #%d sent (%d samples @ %dHz)\n",
                      packetsSent, bufferIndex, OUTPUT_RATE);
    }
}
