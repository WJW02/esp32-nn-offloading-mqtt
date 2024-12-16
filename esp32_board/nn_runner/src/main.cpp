/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* DEPENDENCIES LIBS
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
#include <WiFi.h>
#include <AsyncMqttClient.h>
#include <sys/time.h>
#include <UUID.h>
#include <ArduinoJson.h>
#include "esp_task_wdt.h"

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* LIBS for TFLITE
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"

#define MODEL_NAME "test_model"
#include "model_layers/layers.h"

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*  ARDUINO JSON CONFIGURATION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
struct SpiRamAllocator : ArduinoJson::Allocator {
  void* allocate(size_t size) override {
    return heap_caps_malloc(size, MALLOC_CAP_SPIRAM);
  }

  void deallocate(void* pointer) override {
    heap_caps_free(pointer);
  }

  void* reallocate(void* ptr, size_t new_size) override {
    return heap_caps_realloc(ptr, new_size, MALLOC_CAP_SPIRAM);
  }
};

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*  CONFIGURATIONS & GLOBAL VARIABLES
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
#include "conf.h"

// Communication & Offloading Variables
AsyncMqttClient             mqttClient;
TimerHandle_t               mqttReconnectTimer;
TimerHandle_t               wifiReconnectTimer;
bool                        deviceRegistered = false;
UUID                        uuid;
String                      MessageUUID = "";
String                      DeviceUUID = "";
SpiRamAllocator             allocator;
JsonDocument                jsonDoc(&allocator);

String                      end_computation_topic;
String                      device_registration_topic = "devices/";
String                      model_data_topic;
String                      model_inference_topic;
String                      model_inference_result_topic;

bool                        testFinished = false;
bool                        modelDataLoaded = false;
float*                      inputBuffer = nullptr;
char*                       message = nullptr;
#ifdef FOMO
float*                      lastMultiOutputLayerData = nullptr;
#endif // FOMO

// Neural Network Variables
tflite::MicroErrorReporter  micro_error_reporter;
tflite::ErrorReporter*      error_reporter = &micro_error_reporter;
const tflite::Model*        model = nullptr;
tflite::MicroInterpreter*   interpreter = nullptr;
TfLiteTensor*               input;
TfLiteTensor*               output;
uint8_t*                    tensor_arena = nullptr;
bool                        modelLoaded = false;
bool                        firstInferenceDone = false; 

/*
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* TIMER CONFIGURATION & FLOATING-POINT TIMESTAMP
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
String getCurrTimeStr(){
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  time_t currentTime = tv.tv_sec;
  int milliseconds = tv.tv_usec / 1000;
  int microseconds = tv.tv_usec % 1000000;
  char currentTimeStr[30];
  snprintf(currentTimeStr, sizeof(currentTimeStr), "%ld.%03d%03d", currentTime, milliseconds, microseconds);
  String currentTimeString = String(currentTimeStr);
  return currentTimeString;
}

void timeConfiguration(){
  // Configure NTP time synchronization
  configTime(NTP_GMT_OFFSET, NTP_DAYLIGHT_OFFSET, NTP_SRV);
  Serial.println("Connecting to NTP Server");
  // Try obtaining the time until successful
  struct tm timeinfo;
  while (!getLocalTime(&timeinfo)) {
    delay(500);
  }

  // Print current time
  Serial.println("NTP Time Configured - Current Time: ");
  Serial.println(getCurrTimeStr());
  return;
}

/*
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* PUBLISH DEVICE PREDICTION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void publishDevicePrediction(int offloading_layer_index, JsonArray layer_output, JsonArray layers_inference_time) {    // Generate the JSON message
    jsonDoc["timestamp"] = getCurrTimeStr();
    jsonDoc["message_id"] = MessageUUID;
    jsonDoc["device_id"] = DeviceUUID;
    jsonDoc["message_content"] = JsonObject();
    jsonDoc["message_content"]["layer_output"] = layer_output;
    jsonDoc["message_content"]["offloading_layer_index"] = offloading_layer_index;
    jsonDoc["message_content"]["layers_inference_time"] = layers_inference_time;
    // Serialize the JSON document to a string
    String jsonMessage;
    serializeJson(jsonDoc, jsonMessage);
    // Publish the JSON message to the topic
    mqttClient.publish(model_inference_result_topic.c_str(), 2, false, (char*)jsonMessage.c_str());
    Serial.println("Published Prediction");
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* LOAD NN LAYER
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void loadNeuralNetworkLayer(String layer_name){
  // Import del modello da testare -> Nome nell'header file
  LOAD_LAYER();

  if (model->version() != TFLITE_SCHEMA_VERSION) {
      Serial.println("Model provided is schema version not equal to supported!");
      return;
  } else { Serial.println("Model Layer Loaded!"); }

  // Questo richiama tutte le implementazioni delle operazioni di cui abbiamo bisogno
  tflite::AllOpsResolver resolver;
  tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, K_TENSOR_ARENA_SIZE, error_reporter);
  interpreter = &static_interpreter;
  Serial.println("Interprete ok");

  // Alloco la memoria del tensor_arena per i tensori del modello
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
      Serial.println("AllocateTensors() failed");
      return;
  } else { Serial.println("AllocateTensors() done"); }
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* INFERENCE FOR NN LAYER
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
extern "C" void runNeuralNetworkLayer(int offloading_layer_index, float inputBuffer[]) {
  // Register the current running task to the watchdog
  esp_task_wdt_add(NULL); 

  // Initialize input data with image
  float* inputData = inputBuffer;
  int inputSize = BATCH_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS * sizeof(float);

#ifdef FOMO
  int multiOutputLayers[] = {16, 34, 43};
  size_t multiOutputLayersSize = sizeof(multiOutputLayers)/sizeof(multiOutputLayers[0]);
#endif

  // Assuming inputData is in the format expected by your neural network
  for (int i = 0; i <= offloading_layer_index; i++) {
    String layer_name = "layer_" + String(i);
    float inizio = micros();
    
    loadNeuralNetworkLayer(layer_name); // Load the appropriate layer
    input = interpreter->input(0);

    // Copy the input data to the input tensor
    memcpy(input->data.f, inputData, inputSize);

#ifdef FOMO
    // Copy the input data of last layer with multiple outputs to the input tensor
    if (interpreter->inputs_size() > 1) {
      if (interpreter->inputs_size() == 2) {
        input = interpreter->input(1);
        memcpy(input->data.f, lastMultiOutputLayerData, inputSize);
      } else {
        Serial.println("Error: this device can only handle sequential and FOMO models");
        return;
      }
    }
#endif

#ifdef DEBUG
    Serial.print("LAYER " + String(i) + " INPUT TENSOR: (");
    for (int j = 0; j < input->dims->size; ++j) {
      Serial.print(input->dims->data[j]);
      if (j+1 < input->dims->size) {
        Serial.print(", ");
      }
    }
    Serial.println(")");

    int numInput = 1;
    for (int j = 0; j < input->dims->size; ++j) {
      numInput *= input->dims->data[j];
    }
    Serial.println("LAYER " + String(i) + " INPUT DATA:");
    for (int j = 0; j < numInput; ++j) {
  #ifdef ALL
      Serial.print(inputData[j]);
      Serial.print(" ");
  #else // Only print first 3 and last 3 elements
      if (j < 3) {
        Serial.print(inputData[j]);
        Serial.print(" ");
      } else if (j >= numInput-3) {
        Serial.print(inputData[j]);
        Serial.print(" ");
      }
  #endif // ALL
    }
    Serial.println();
  #ifdef FOMO
    if (interpreter->inputs_size() == 2) {
      Serial.println("LAYER " + String(i) + " INPUT DATA 2:");
      for (int j = 0; j < numInput; ++j) {
    #ifdef ALL
        Serial.print(lastMultiOutputLayerData[j]);
        Serial.print(" ");
    #else // Only print first 3 and last 3 elements
        if (j < 3) {
          Serial.print(lastMultiOutputLayerData[j]);
          Serial.print(" ");
        } else if (j >= numInput-3) {
          Serial.print(lastMultiOutputLayerData[j]);
          Serial.print(" ");
        }
    #endif // ALL
      }
      Serial.println();
    }
  #endif // FOMO
#endif // DEBUG

    // Run inference
    esp_task_wdt_reset();
    interpreter->Invoke();
    esp_task_wdt_reset();

    // Extract relevant information from the output tensor
    output = interpreter->output(0);
    float* outputData = output->data.f;
    int outputSize = output->bytes;

#ifdef DEBUG
    Serial.print("LAYER " + String(i) + " OUTPUT TENSOR: (");
    for (int j = 0; j < output->dims->size; ++j) {
      Serial.print(output->dims->data[j]);
      if (j+1 < output->dims->size) {
        Serial.print(", ");
      }
    }
    Serial.println(")");

    int numOutput = 1;
    for (int j = 0; j < output->dims->size; ++j) {
      numOutput *= output->dims->data[j];
    }
    Serial.println("LAYER " + String(i) + " OUTPUT DATA:");
    for (int j = 0; j < numOutput; ++j) {
  #ifdef ALL
      Serial.print(outputData[j]);
      Serial.print(" ");
  #else // Only print first 3 and last 3 elements
      if (j < 3) {
        Serial.print(outputData[j]);
        Serial.print(" ");
      } else if (j >= numOutput-3) {
        Serial.print(outputData[j]);
        Serial.print(" ");
      }
  #endif // ALL
    }
    Serial.println();
#endif // DEBUG

    // Set next layer's input data and size
    memcpy(inputData, outputData, outputSize);
    inputSize = outputSize;

#ifdef FOMO
    // Saves the output of the layer with multiple outputs for future use
    for (int j = 0; j < multiOutputLayersSize; ++j) {
      if (i == multiOutputLayers[j]) {
        memcpy(lastMultiOutputLayerData, outputData, outputSize);
        break;
      }
    }
#endif

    // Store inference time in seconds
    jsonDoc["layer_inference_time"][i] = (micros() - inizio) / 1000000.0; // Convert microseconds to seconds

    Serial.println("Computed layer: " + String(i) + " Inf Time: " + String((micros() - inizio) / 1000000.0) + " s");
  }

  // Compute number of elements
  int numOutput = 1;
  for (int i = 0; i < output->dims->size; ++i) {
    numOutput *= output->dims->data[i];
  }

  // Store offloading layer output
  float* outputData = output->data.f;
  for (int i = 0; i < numOutput; ++i) {
    jsonDoc["layer_output"][i] = outputData[i];
  }

  Serial.println("Last layer output: "+jsonDoc["layer_output"].as<String>());
  publishDevicePrediction(offloading_layer_index, jsonDoc["layer_output"], jsonDoc["layer_inference_time"]);
}


/*
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * GENERATE MESSAGE UUID
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void generateMessageUUID(){
  // Generate a UUID
  unsigned long seed = esp_random();
  uuid.seed(seed);
  uuid.setRandomMode();
  uuid.generate();
  MessageUUID = (String)uuid.toCharArray();
  MessageUUID = MessageUUID.substring(0, 4);
  DeviceUUID = "device_01"; // + MessageUUID;
}

/*
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* REGISTER THE DEVICE ON THE EDGE
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void registerDevice(){
  // Generate the JSON message
  jsonDoc["timestamp"] = getCurrTimeStr();
  jsonDoc["message_id"] = MessageUUID;
  jsonDoc["device_id"] = DeviceUUID;
  jsonDoc["message_content"] = "HelloWorld!";
  // Serialize the JSON document to a string
  String jsonMessage;
  serializeJson(jsonDoc, jsonMessage);
  // Publish the JSON message to the topic
  mqttClient.publish(device_registration_topic.c_str(), 2, false, (char*)jsonMessage.c_str());
  Serial.println("Device Registered");
  deviceRegistered = true;
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* GET MODEL DATA FOR PREDICTION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
#include <ArduinoJson.h>  // Make sure to include the ArduinoJson library

void getModelDataForPrediction(const JsonArray& inputData) {
  // Convert the inputData string to a 2D array
  try {
    for (int b = 0; b < BATCH_SIZE; ++b) {
      for (int h = 0; h < IMAGE_HEIGHT; ++h) {
        for (int w = 0; w < IMAGE_WIDTH; ++w) {
          for (int c = 0; c < CHANNELS; ++c) {
            int i = b*IMAGE_HEIGHT*IMAGE_WIDTH*CHANNELS + h*IMAGE_WIDTH*CHANNELS + w*CHANNELS + c;
            inputBuffer[i] = inputData[b][h][w][c].as<float>() / 255.0; // Assuming inputData contains numeric characters
          }
        }
      }
    }
    Serial.println("Model input data received");
  } catch (const std::exception& e) {
    Serial.print("Error receiving model input data: ");
    Serial.println(e.what());
  }
  modelDataLoaded = true;
}

void processIncomingMessage(char* topic, char* payload, AsyncMqttClientMessageProperties properties, size_t len, size_t index, size_t total) {
  // Instantiate message
  if (index == 0) {
    message = new char[total+1];
  }

  // Bufferize message fragments
  for (size_t i = 0; i < len; i++) {
    message[index+i] = payload[i];
  }

  // Return if we don't have whole message yet
  if (index+len < total) {
    return;
  }

  // Add terminator
  message[total] = '\0';

  // Parse the JSON message and store it in the DynamicJsonDocument
  JsonDocument doc(&allocator);
  DeserializationError error = deserializeJson(doc, message);

  // Check for parsing errors
  if (error) {
    Serial.print("JSON parsing error: ");
    Serial.println(error.c_str());
    delete[] message;
    message = nullptr;
    return;
  }
  Serial.print("Received message:");
  Serial.println(topic);

  // Check if the message is for model_data
  if (strcmp(topic, model_data_topic.c_str()) == 0) {
    JsonArray inputData = doc["input_data"]; 
    getModelDataForPrediction(inputData);
  }
  
  // Check if the message is for model_inference
  if (strcmp(topic, model_inference_topic.c_str()) == 0) {
    int offloading_layer_index = doc["offloading_layer_index"];
    if (offloading_layer_index >= MAX_NUM_LAYER) {
      Serial.println("Invalid offloading layer");
      return;
    }
    JsonArray inputData = doc["input_data"]; 
    getModelDataForPrediction(inputData);
    runNeuralNetworkLayer(offloading_layer_index, inputBuffer);
  }

  // Check if the test is finished
  if (strcmp(topic, end_computation_topic.c_str()) == 0) {
    Serial.println("Ending Computation");
    testFinished = true;
  }

  delete[] message;
  message = nullptr;
}

void dispatchCallbackMessages() {
  // Set the topics
  end_computation_topic = DeviceUUID + "/end_computation";
  model_data_topic = DeviceUUID + "/model_data";
  model_inference_topic = DeviceUUID + "/model_inference";
  model_inference_result_topic = DeviceUUID + "/model_inference_result";

  // Subscribe to the topic
  mqttClient.subscribe(model_data_topic.c_str(), 2);
  mqttClient.subscribe(model_inference_topic.c_str(), 2);
  mqttClient.subscribe(end_computation_topic.c_str(), 2);

  // Set the callback function
  mqttClient.onMessage(processIncomingMessage);
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* MQTT CONFIGURATION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void connectToMqtt() {
  Serial.println("Connecting to MQTT...");
  mqttClient.connect();
}

void onMqttConnect(bool sessionPresent) {
  Serial.println("Connected to MQTT.");
  timeConfiguration();          // Synchronize Timer - NTP server
  generateMessageUUID();        // Generate an Identifier for the message
  dispatchCallbackMessages();   // Set the callback function for the MQTT messages
  registerDevice();             // Register the device on the edge
}

void onMqttDisconnect(AsyncMqttClientDisconnectReason reason) {
  Serial.println("Disconnected form MQTT.");
  if (WiFi.isConnected()) {
    xTimerStart(mqttReconnectTimer, 0);
  }
}

void mqttConfiguration(){
  mqttClient.onConnect(onMqttConnect);
  mqttClient.onDisconnect(onMqttDisconnect);
  mqttClient.setServer(MQTT_SRV, MQTT_PORT);
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* WIFI CONFIGURATION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void connectToWifi() {
  Serial.println("Connecting to WiFi...");
  WiFi.begin(SSID, PWD);
}

void WiFiEvent(WiFiEvent_t event) {
  switch(event) {
    case ARDUINO_EVENT_WIFI_STA_GOT_IP:
      Serial.println("WiFi connected");
      Serial.println("IP address: ");
      Serial.println(WiFi.localIP());
      connectToMqtt();
      break;
    case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
      Serial.println("WiFi lost connection");
      xTimerStop(mqttReconnectTimer, 0); // ensures we don't reconnect to MQTT while reconnecting to WiFi
      xTimerStart(wifiReconnectTimer, 0);
      break;
  }
}

void wifiConfiguration(){
  WiFi.onEvent(WiFiEvent);
}

/*
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * SETUP 
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 */
void setup() {
  Serial.begin(115200);
  esp_task_wdt_init(WDT_TIMEOUT, true); // Watchdog Timer
  if (psramInit()) {
    Serial.println("The PSRAM is correctly initialized");
  } else {
    Serial.println("PSRAM does not work");
  }
  inputBuffer = (float*)ps_malloc(MAX_ELEMENTS_PER_MODEL_LAYER*sizeof(float)); 
  tensor_arena = (uint8_t*)ps_malloc(K_TENSOR_ARENA_SIZE*sizeof(uint8_t));
#ifdef FOMO
  lastMultiOutputLayerData = (float*)ps_malloc(MAX_ELEMENTS_PER_MODEL_LAYER*sizeof(float)); 
#endif // FOMO
  mqttReconnectTimer = xTimerCreate("mqttTimer", pdMS_TO_TICKS(2000), pdFALSE, (void*)0, reinterpret_cast<TimerCallbackFunction_t>(connectToMqtt));
  wifiReconnectTimer = xTimerCreate("wifiTimer", pdMS_TO_TICKS(2000), pdFALSE, (void*)0, reinterpret_cast<TimerCallbackFunction_t>(connectToWifi));
  wifiConfiguration();          // Wi-Fi Connection
  mqttConfiguration();          // MQTT
  connectToWifi();
}

/* 
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * LOOP 
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 */
void loop() {
  if(testFinished){
    delay(10000);
    ESP.restart();
  }
}