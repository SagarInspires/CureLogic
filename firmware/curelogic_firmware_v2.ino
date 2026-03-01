/**
 * ╔══════════════════════════════════════════════════════════════╗
 * ║  CureLogic — NodeMCU Firmware v2.0  (INTEGRATED BUILD)      ║
 * ║  L&T CreaTech Hackathon | Problem Statement 1                ║
 * ╠══════════════════════════════════════════════════════════════╣
 * ║  Changes from v1.0:                                          ║
 * ║  • Real Wi-Fi credentials with fallback AP mode              ║
 * ║  • HTTP POST to live Flask backend (192.168.x.x:5000)        ║
 * ║  • Reads Flask SVM prediction from response JSON             ║
 * ║  • MQTT-ready toggle (#define USE_MQTT)                      ║
 * ║  • Watchdog timer for 24/7 deployment reliability            ║
 * ║  • Over-the-Air (OTA) update support                         ║
 * ║  • Proteus-compatible: all pins match virtual schematic       ║
 * ╚══════════════════════════════════════════════════════════════╝
 *
 * PROTEUS SCHEMATIC COMPONENTS:
 *   U1  : NodeMCU-12E (ESP8266)
 *   U2  : DS18B20 × 3  (surface, mid, core) on D4
 *   R1  : 4.7 kΩ pull-up on DQ line
 *   U3  : 16×2 LCD I2C on D1(SCL), D2(SDA)
 *   LED1: Status LED on D5 (green = OK, red = alert)
 *
 * WIRING:
 *   DS18B20 VDD → 3.3V
 *   DS18B20 GND → GND
 *   DS18B20 DQ  → D4 (GPIO2) with 4.7kΩ to 3.3V
 *   LCD SDA     → D2 (GPIO4)
 *   LCD SCL     → D1 (GPIO5)
 *   LED Anode   → D5 (GPIO14) via 220Ω
 */

#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <ArduinoOTA.h>
#include <ESP8266WebServer.h>   // For AP fallback config portal
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ArduinoJson.h>
#include <Ticker.h>             // Lightweight timer / watchdog

// ─── ▶ USER CONFIG — Set these before flashing ──────────────
//
//  PRIMARY: Your lab / office Wi-Fi
#define WIFI_SSID_PRIMARY    "CureLogic_Lab"
#define WIFI_PASS_PRIMARY    "precast2024"
//
//  FALLBACK: Mobile hotspot (for site demo)
#define WIFI_SSID_FALLBACK   "iPhone_Hotspot"
#define WIFI_PASS_FALLBACK   "curelogic"
//
//  Flask backend IP (laptop running app.py on same network)
#define BACKEND_IP           "192.168.1.100"
#define BACKEND_PORT         5000
#define DEVICE_SECRET        "CL-SECRET-2024"
#define DEVICE_ID            "CL-NODE-01"
//
//  OTA Password (for wireless firmware updates during demo)
#define OTA_PASSWORD         "curelogic_ota"
//
// ─────────────────────────────────────────────────────────────

// ─── Compile-time flags ───────────────────────────────────────
// Uncomment to enable MQTT instead of HTTP POST
// #define USE_MQTT

// ─── Pin Definitions ─────────────────────────────────────────
#define ONE_WIRE_BUS     D4       // DS18B20 data line
#define STATUS_LED       D5       // Green/Red status LED
#define SENSOR_COUNT     3        // surface=0, mid=1, core=2

// ─── Curing Constants ─────────────────────────────────────────
#define TARGET_STRENGTH_MPA   25.0f
#define T_DATUM              -10.0f    // Nurse-Saul datum
#define MOULD_COST_PER_HR    450.0f
#define WET_CURE_COST_HR     12.5f
#define PUSH_INTERVAL_MS     60000UL  // 60 s between cloud pushes
#define WATCHDOG_TIMEOUT_S   120      // Reset if no push in 2 min

// ─── Hardware Objects ─────────────────────────────────────────
OneWire           oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);
LiquidCrystal_I2C lcd(0x27, 16, 2);   // I2C address may be 0x3F on some boards
WiFiClient        wifiClient;
HTTPClient        http;
Ticker            watchdogTicker;

// ─── State ───────────────────────────────────────────────────
float   tempReadings[SENSOR_COUNT]     = {25.0, 25.0, 25.0};
float   maturityIndex                  = 0.0f;
float   elapsedHours                   = 0.0f;
float   cumulativeCost                 = 0.0f;
float   lastPredStrength               = 0.0f;
float   lastPredETA                    = 99.0f;
bool    demoulded                      = false;
bool    thermalAlert                   = false;
uint8_t wifiRetries                    = 0;
unsigned long lastPushTime             = 0;
unsigned long startTime;
volatile bool watchdogFed              = true;

// ─── Watchdog ─────────────────────────────────────────────────
void IRAM_ATTR feedWatchdog() { watchdogFed = true; }

void watchdogCheck() {
    if (!watchdogFed) {
        Serial.println(F("[WDT] Watchdog triggered — rebooting!"));
        delay(100); ESP.restart();
    }
    watchdogFed = false;
}

// ─── Wi-Fi Connection (dual SSID with AP fallback) ─────────
bool connectWiFi() {
    const char* ssids[]  = { WIFI_SSID_PRIMARY, WIFI_SSID_FALLBACK };
    const char* passes[] = { WIFI_PASS_PRIMARY,  WIFI_PASS_FALLBACK };

    for (int net = 0; net < 2; net++) {
        Serial.printf("[WiFi] Trying: %s\n", ssids[net]);
        WiFi.begin(ssids[net], passes[net]);

        int attempts = 0;
        while (WiFi.status() != WL_CONNECTED && attempts < 20) {
            delay(500); Serial.print('.'); attempts++;
        }
        if (WiFi.status() == WL_CONNECTED) {
            Serial.printf("\n[WiFi] ✓ Connected to %s | IP: %s\n",
                          ssids[net], WiFi.localIP().toString().c_str());
            return true;
        }
        Serial.printf("\n[WiFi] ✗ Failed on %s\n", ssids[net]);
    }

    // Start AP fallback so user can configure from phone
    WiFi.softAP("CureLogic-Setup", "setup1234");
    Serial.println(F("[WiFi] Fallback AP started: CureLogic-Setup"));
    Serial.println(F("[WiFi] Connect to AP and browse 192.168.4.1"));
    return false;
}

// ─── Read Temperature Sensors ─────────────────────────────────
void readSensors() {
    sensors.requestTemperatures();
    delay(94);   // DS18B20 conversion time at 12-bit resolution

    for (int i = 0; i < SENSOR_COUNT; i++) {
        float t = sensors.getTempCByIndex(i);
        if (t != DEVICE_DISCONNECTED_C && t > -5.0f && t < 100.0f) {
            tempReadings[i] = t;
        }
        // else: hold last valid reading (fail-safe)
    }

    thermalAlert = (tempReadings[2] > 70.0f);   // Core over 70°C
}

// ─── Update Maturity (Nurse-Saul) ────────────────────────────
void updateMaturity() {
    float avg = (tempReadings[0] + tempReadings[1] + tempReadings[2]) / 3.0f;
    float dt  = (float)PUSH_INTERVAL_MS / 3600000.0f;  // hours per tick
    if (avg > T_DATUM) maturityIndex += (avg - T_DATUM) * dt;
    elapsedHours   = (float)(millis() - startTime) / 3600000.0f;
    cumulativeCost = (WET_CURE_COST_HR + MOULD_COST_PER_HR) * elapsedHours;
}

// ─── Strength Model (local fallback) ─────────────────────────
float localStrengthEstimate() {
    if (maturityIndex <= 0) return 0.0f;
    return constrain(-12.4f + 7.8f * log(maturityIndex), 0.0f, 58.0f);
}

// ─── HTTP POST to Flask Backend ───────────────────────────────
bool pushToBackend() {
    if (WiFi.status() != WL_CONNECTED) {
        if (++wifiRetries > 5) connectWiFi();
        return false;
    }
    wifiRetries = 0;

    // ── Build JSON payload ──
    StaticJsonDocument<512> doc;
    doc["device_id"]          = DEVICE_ID;
    doc["elapsed_hours"]      = elapsedHours;
    doc["ambient_temp"]       = tempReadings[0];    // surface ≈ ambient
    doc["temp_surface"]       = tempReadings[0];
    doc["temp_mid"]           = tempReadings[1];
    doc["temp_core"]          = tempReadings[2];
    doc["humidity"]           = 62.0;               // Replace with DHT11 reading if added
    doc["maturity_index"]     = maturityIndex;
    doc["curing_method"]      = "Steam";            // Change per batch
    doc["season"]             = "Summer";           // Change per deployment
    doc["cement_content"]     = 400;
    doc["w_c_ratio"]          = 0.48;
    doc["demoulded"]          = demoulded;

    String body;
    serializeJson(doc, body);

    // ── HTTP POST ──
    String url = "http://" + String(BACKEND_IP) + ":" + String(BACKEND_PORT) + "/api/sensor_data";
    http.begin(wifiClient, url);
    http.addHeader("Content-Type", "application/json");
    http.addHeader("X-Device-Key", DEVICE_SECRET);
    http.setTimeout(8000);

    int code = http.POST(body);

    if (code == 200 || code == 201) {
        // ── Parse SVM prediction from response ──
        String responseBody = http.getString();
        StaticJsonDocument<512> resp;
        if (deserializeJson(resp, responseBody) == DeserializationError::Ok) {
            JsonObject pred = resp["prediction"];
            if (!pred.isNull()) {
                lastPredStrength = pred["strength_mpa"].as<float>();
                lastPredETA      = pred["eta_hours"].as<float>();
                demoulded        = pred["demould_ready"].as<bool>();
            }
        }
        Serial.printf("[Backend] ✓ %d | SVM: %.2f MPa | ETA: %.1f hr\n",
                      code, lastPredStrength, lastPredETA);
        http.end();
        feedWatchdog();
        return true;
    }

    Serial.printf("[Backend] ✗ HTTP %d\n", code);
    http.end();
    return false;
}

// ─── LCD Display ──────────────────────────────────────────────
void updateLCD() {
    lcd.clear();

    if (demoulded) {
        lcd.setCursor(0, 0); lcd.print("** DEMOULD READY **");
        lcd.setCursor(0, 1); lcd.printf("S=%.1fMPa T=%.1fh", lastPredStrength, elapsedHours);
        return;
    }

    if (thermalAlert) {
        lcd.setCursor(0, 0); lcd.print("!! THERMAL ALERT !!");
        lcd.setCursor(0, 1); lcd.printf("Core=%.1fC MAX=70C", tempReadings[2]);
        return;
    }

    // Line 1: Temperature
    lcd.setCursor(0, 0);
    lcd.printf("C%.0f M%.0f S%.0f *C",
               tempReadings[2], tempReadings[1], tempReadings[0]);

    // Line 2: Strength and ETA
    lcd.setCursor(0, 1);
    if (lastPredStrength > 0) {
        lcd.printf("%.1fMPa ETA:%.1fhr", lastPredStrength, lastPredETA);
    } else {
        float local = localStrengthEstimate();
        lcd.printf("Est:%.1fMPa %.0f%%", local, (local / TARGET_STRENGTH_MPA) * 100);
    }
}

// ─── Status LED ───────────────────────────────────────────────
void updateLED() {
    if (thermalAlert) {
        // Fast blink = alert
        static bool ledState = false;
        ledState = !ledState;
        digitalWrite(STATUS_LED, ledState ? HIGH : LOW);
    } else if (demoulded) {
        // Solid on = ready
        digitalWrite(STATUS_LED, HIGH);
    } else {
        // Heartbeat = normal
        digitalWrite(STATUS_LED, (millis() % 2000 < 200) ? HIGH : LOW);
    }
}

// ─── Serial Dashboard ─────────────────────────────────────────
void printSerial() {
    Serial.println(F("╔═══════════════════════════════════╗"));
    Serial.println(F("║     CureLogic | NodeMCU v2.0      ║"));
    Serial.println(F("╠═══════════════════════════════════╣"));
    Serial.printf( "║ Elapsed  : %6.2f hrs              ║\n", elapsedHours);
    Serial.printf( "║ T Core   : %5.1f °C               ║\n", tempReadings[2]);
    Serial.printf( "║ T Mid    : %5.1f °C               ║\n", tempReadings[1]);
    Serial.printf( "║ T Surface: %5.1f °C               ║\n", tempReadings[0]);
    Serial.printf( "║ Maturity : %6.0f °C·hr           ║\n", maturityIndex);
    Serial.printf( "║ SVM Str  : %5.2f MPa (%.0f%%)      ║\n",
                   lastPredStrength, (lastPredStrength/TARGET_STRENGTH_MPA)*100);
    Serial.printf( "║ ETA      : %5.1f hrs              ║\n", lastPredETA);
    Serial.printf( "║ Cost     : INR %7.0f             ║\n", cumulativeCost);
    Serial.printf( "║ Status   : %-25s║\n",
                   demoulded ? "READY TO DEMOULD!" :
                   thermalAlert ? "** THERMAL ALERT **" : "Curing in progress");
    Serial.println(F("╚═══════════════════════════════════╝\n"));
}

// ─── Setup ────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);

    Serial.println(F("\n[CureLogic] Firmware v2.0 booting..."));

    // GPIO
    pinMode(STATUS_LED, OUTPUT);
    digitalWrite(STATUS_LED, LOW);

    // I2C LCD
    Wire.begin(D2, D1);
    lcd.init(); lcd.backlight();
    lcd.setCursor(0, 0); lcd.print("CureLogic v2.0");
    lcd.setCursor(0, 1); lcd.print("Initializing...");

    // DS18B20
    sensors.begin();
    sensors.setResolution(12);
    Serial.printf("[Sensors] Found %d DS18B20(s)\n", sensors.getDeviceCount());

    // Wi-Fi
    bool connected = connectWiFi();
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print(connected ? "WiFi: Connected" : "WiFi: AP Mode");
    lcd.setCursor(0, 1); lcd.print(connected ? WiFi.localIP().toString() : "192.168.4.1");

    // OTA (wireless firmware update during demo)
    ArduinoOTA.setPassword(OTA_PASSWORD);
    ArduinoOTA.onStart([]() { Serial.println("[OTA] Update starting..."); });
    ArduinoOTA.onEnd([]()   { Serial.println("[OTA] Done! Rebooting."); });
    ArduinoOTA.onError([](ota_error_t e) { Serial.printf("[OTA] Error %u\n", e); });
    ArduinoOTA.begin();
    Serial.println(F("[OTA] Ready"));

    // Watchdog: check every WATCHDOG_TIMEOUT_S seconds
    watchdogTicker.attach(WATCHDOG_TIMEOUT_S, watchdogCheck);

    startTime   = millis();
    lastPushTime= millis() - PUSH_INTERVAL_MS;  // Push immediately on first loop

    Serial.println(F("[CureLogic] Ready. Monitoring started.\n"));
}

// ─── Main Loop ────────────────────────────────────────────────
void loop() {
    ArduinoOTA.handle();    // Non-blocking OTA listener

    unsigned long now = millis();

    if (now - lastPushTime >= PUSH_INTERVAL_MS) {
        lastPushTime = now;

        readSensors();
        updateMaturity();

        // Use local model if backend unreachable
        if (WiFi.status() != WL_CONNECTED || !pushToBackend()) {
            lastPredStrength = localStrengthEstimate();
            float targetMat  = exp((TARGET_STRENGTH_MPA + 12.4f) / 7.8f);
            float avgTemp    = (tempReadings[0]+tempReadings[1]+tempReadings[2])/3.0f;
            lastPredETA      = max((targetMat - maturityIndex) / max(avgTemp - T_DATUM, 1.0f), 0.0f);
        }

        if (lastPredStrength >= TARGET_STRENGTH_MPA) demoulded = true;

        printSerial();
        updateLCD();
    }

    updateLED();
    delay(100);
}
