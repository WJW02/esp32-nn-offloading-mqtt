#ifndef CONF_H_
#define CONF_H_

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// WiFi Conf Hotspot
const char* SSID                = "";
const char* PWD                 = "";
// ------------------------------------------------------------------------------------------------------------------------------------------------------
// MQTT Conf
const char* MQTT_SRV            = "hostaname.local"; // .local needed when using Hotspot so i will leave it by default
const int MQTT_PORT             = 1883;
const char* MQTT_USR            = ""; 
const char* MQTT_PWD            = ""; 
#define MQTT_MAX_PACKET_SIZE 24*1024
// ------------------------------------------------------------------------------------------------------------------------------------------------------
// NTP Timer Configuration
const char* NTP_SRV             = "0.it.pool.ntp.org";
const long NTP_GMT_OFFSET       = 0;
const int NTP_DAYLIGHT_OFFSET   = 0;
// ------------------------------------------------------------------------------------------------------------------------------------------------------
// JSON Document Size Configuration
constexpr int INPUT_JSONDOC_SIZE = 65*1024;
constexpr int OUTPUT_JSONDOC_SIZE = 65*1024;
// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Model Configuration
const int MAX_NUM_LAYER = 7;
constexpr int BATCH_SIZE = 1;
constexpr int IMAGE_HEIGHT = 32;
constexpr int IMAGE_WIDTH = 32;
constexpr int CHANNELS = 3;
constexpr int MAX_ELEMENTS_PER_MODEL_LAYER = 1*32*32*16;
constexpr int K_TENSOR_ARENA_SIZE = 85*1024;
// ------------------------------------------------------------------------------------------------------------------------------------------------------

#endif // CONF_H
