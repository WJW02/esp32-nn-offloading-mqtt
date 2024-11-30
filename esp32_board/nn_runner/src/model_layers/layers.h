#include "layer_0.h"
#include "layer_1.h"
#include "layer_2.h"
#include "layer_3.h"
#include "layer_4.h"
#include "layer_5.h"
#include "layer_6.h"
#define LOAD_LAYER() if(layer_name.equals("layer_0"))model = tflite::GetModel(layer_0);\
if(layer_name.equals("layer_1"))model = tflite::GetModel(layer_1);\
if(layer_name.equals("layer_2"))model = tflite::GetModel(layer_2);\
if(layer_name.equals("layer_3"))model = tflite::GetModel(layer_3);\
if(layer_name.equals("layer_4"))model = tflite::GetModel(layer_4);\
if(layer_name.equals("layer_5"))model = tflite::GetModel(layer_5);\
if(layer_name.equals("layer_6"))model = tflite::GetModel(layer_6);\
