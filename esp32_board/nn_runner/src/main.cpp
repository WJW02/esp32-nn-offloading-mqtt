/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* DEPENDENCIES LIBS
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
#include <WiFi.h>
#include <PubSubClient.h>
#include <sys/time.h>
#include <UUID.h>
#include <ArduinoJson.h>

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
*  CONFIGURATIONS & GLOBAL VARIABLES
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
#include "conf.h"

// Communication & Offloading Variables
WiFiClient                  espClient;
PubSubClient                client(espClient);
bool                        deviceRegistered = false;
UUID                        uuid;
String                      MessageUUID = "";
String                      DeviceUUID = "";
StaticJsonDocument<OUTPUT_JSONDOC_SIZE> jsonDoc;

String                      end_computation_topic;
String                      device_registration_topic = "devices/";
String                      model_data_topic;
String                      model_inference_topic;
String                      model_inference_result_topic;

bool                        testFinished = false;
bool                        modelDataLoaded = false;
float                       inputBuffer[MAX_ELEMENTS_PER_MODEL_LAYER] = {};

// Neural Network Variables
tflite::MicroErrorReporter  micro_error_reporter;
tflite::ErrorReporter*      error_reporter = &micro_error_reporter;
const tflite::Model*        model = nullptr;
tflite::MicroInterpreter*   interpreter = nullptr;
TfLiteTensor*               input;
TfLiteTensor*               output;
uint8_t                     tensor_arena[K_TENSOR_ARENA_SIZE];
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
    client.publish(model_inference_result_topic.c_str(), jsonMessage.c_str(), 2);
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
  // Initialize input data with image
  float* inputData = inputBuffer;
  int inputSize = BATCH_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS * sizeof(float);

  // Assuming inputData is in the format expected by your neural network
  for (int i = 0; i <= offloading_layer_index; i++) {
    String layer_name = "layer_" + String(i);
    float inizio = micros();
    
    loadNeuralNetworkLayer(layer_name); // Load the appropriate layer
    input= interpreter->input(0);

    // Copy the input data to the input tensor
    memcpy(input->data.f, inputData, inputSize);

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
      Serial.print(inputData[j]);
      Serial.print(" ");
    }
    Serial.println();
#endif // DEBUG

    // Run inference
    interpreter->Invoke();

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
      Serial.print(outputData[j]);
      Serial.print(" ");
    }
    Serial.println();
#endif // DEBUG

    // Set next layer's input data and size
    memcpy(inputData, outputData, outputSize);
    inputSize = outputSize;

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
* WIFI CONFIGURATION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void wifiConfiguration(){
  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PWD);
  Serial.println("Connecting to WiFi...");
  while (WiFi.waitForConnectResult() != WL_CONNECTED) {
    Serial.println(".");
    delay(500);
    ESP.restart();
  } 
  Serial.println("Connected to WiFi - IP Address: ");
  Serial.println(WiFi.localIP());
  delay(500);
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* MQTT CONFIGURATION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void mqttConfiguration(){
  client.setServer(MQTT_SRV, MQTT_PORT);
  while (!client.connect("ESP32Client", "", "")) {
    Serial.println("Connecting to MQTT Broker");
    if (!client.connected()) {
      Serial.println("Failed to connect to MQTT Broker - retrying, rc=");
      Serial.println(client.state());
      delay(500);
    }
  }
  client.setBufferSize(MQTT_MAX_PACKET_SIZE);
  Serial.println("Connected to MQTT Broker");
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
  client.publish(device_registration_topic.c_str(), jsonMessage.c_str(), 2);
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
            inputBuffer[i] = inputData[b][h][w][c]; // Assuming inputData contains numeric characters
          }
        }
      }
    }
    Serial.println();
    Serial.println("Model input data received");
  } catch (const std::exception& e) {
    Serial.print("Error receiving model input data: ");
    Serial.println(e.what());
  }
  modelDataLoaded = true;
}

void processIncomingMessage(char* topic, byte* payload, unsigned int length) {
  // Convert the incoming message to a string
  String message;
  message.reserve(length); // Reserve space in advance for efficiency
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  // Parse the JSON message and store it in the DynamicJsonDocument
  DynamicJsonDocument doc(INPUT_JSONDOC_SIZE);
  DeserializationError error = deserializeJson(doc, message);

  // Check for parsing errors
  if (error) {
    Serial.print("JSON parsing error: ");
    Serial.println(error.c_str());
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
    Serial.print("Ending Computation");
    testFinished = true;
  }
}

void dispatchCallbackMessages() {
  // Set the topics
  end_computation_topic = DeviceUUID + "/end_computation";
  model_data_topic = DeviceUUID + "/model_data";
  model_inference_topic = DeviceUUID + "/model_inference";
  model_inference_result_topic = DeviceUUID + "/model_inference_result";

  // Subscribe to the topic
  client.subscribe(model_data_topic.c_str());
  client.subscribe(model_inference_topic.c_str());
  client.subscribe(end_computation_topic.c_str());

  // Set the callback function
  client.setCallback([](char* topic, byte* payload, unsigned int length) {
    processIncomingMessage(topic, payload, length);
  });
}

/*
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * SETUP 
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 */
void setup() {
  Serial.begin(115200);
  wifiConfiguration();          // Wi-Fi Connection
  mqttConfiguration();          // MQTT
  timeConfiguration();          // Synchronize Timer - NTP server
  generateMessageUUID();        // Generate an Identifier for the message
  dispatchCallbackMessages();   // Set the callback function for the MQTT messages
  registerDevice();             // Register the device on the edge
}

/* 
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * LOOP 
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 */
void loop() {
  client.loop(); 
  if(testFinished){
    delay(10000);
    ESP.restart();
  }
}