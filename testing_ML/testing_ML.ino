#include <TensorFlowLite.h>

/**
   Test sinewave neural network model

   Author: Pete Warden
   Modified by: Shawn Hymel
   Date: March 11, 2020

   Copyright 2019 The TensorFlow Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

// Import TensorFlow stuff
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


// Our model
#include "PCD_Model.h"

// Figure out what's going on in our model
#define DEBUG 0

// Some settings

// TFLite globals, used for compatibility with Arduino-style sketches
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// Create an area of memory to use for input, output, and other TensorFlow
// arrays. You'll need to adjust this by combiling, running, and looking
// for errors.
constexpr int kTensorArenaSize = 8 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
} // namespace

void setup() {
  // Wait for Serial to connect

#if DEBUG
  while (!Serial);
#endif

  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(PCD_Model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while (1);
  }

  // Pull in only needed operations (should match NN layers)
  // Available ops:
  //  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/micro_ops.h
  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    while (1);
  }

  // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  // Get information about the memory area to use for the model's input
  // Supported data types:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h#L226
#if DEBUG
  Serial.print("Number of dimensions: ");
  Serial.println(model_input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(model_input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(model_input->dims->data[1]);
  Serial.print("Input type: ");
  Serial.println(model_input->type);
#endif
}




void loop() {
  float duration = millis();
#if DEBUG
  unsigned long start_timestamp = micros();
#endif

  // Calculate x value to feed to the model
  float x_val = 1;


  // Copy value to input buffer (tensor)
  model_input->data.f[0] = -0.029;
  model_input->data.f[1] = -0.059222;
  model_input->data.f[2] = -0.156444;
  model_input->data.f[3] = 0.5005;
  model_input->data.f[4] = 0.069222;
  model_input->data.f[5] = 0.182833;
  model_input->data.f[6] = -0.055056;
  model_input->data.f[7] = -0.148278;
  model_input->data.f[8] = 0.084944;
  model_input->data.f[9] = 0.786556;
  model_input->data.f[10] = -0.194778;
  model_input->data.f[11] = 0.059444;

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on input: %f\n", x_val);
  }

  // Read predicted y value from output buffer (tensor)
  float y_val[5];
  y_val[0] = model_output->data.f[0];
  y_val[1] = model_output->data.f[1];
  y_val[2] = model_output->data.f[2];
  y_val[3] = model_output->data.f[3];
  y_val[4] = model_output->data.f[4];

    for(int i=0;i<5;i++){
      Serial.println(y_val[i]);
    }
  print_y_val(y_val);
  Serial.print("Dur: ");
  Serial.println(millis() - duration);
}

void print_y_val(float y_val[]) {
  int val_len = sizeof(y_val) / sizeof(float);
  Serial.print("Length: ");
  Serial.println(val_len);
  float denom = 0;
  for (int i = 0; i < 5; i++) {
    denom += exp(y_val[i]);
  }
  Serial.println(denom);
  for (int i = 0; i < 5; i++) {
    y_val[i]=(exp(y_val[i]))/denom;
    Serial.println(y_val[i]);
  }
}
