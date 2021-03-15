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


/*
    Name:       Arduino MPU6050 Polling Test.ino
    Created:  10/5/2019 1:31:08 PM
    Author:     FRANKWIN10\Frank

  This is a complete, working MPU6050 (GY-521) example using polling vs interrupts.
  It is based on Jeff Rowberg's MPU6050_DMP6_using_DMP_V6.12 example.  After confirming
  that the example worked properly using interrupts, I modified it to remove the need
  for the interrupt line.  
*/

#include <TensorFlowLite.h>
#include<stdlib.h>
#include "I2Cdev.h"
#include "MPU6050V6.h"  

// Import TensorFlow stuff
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

//We are using a polling approach by toggling different IMUs to possess address 0x69
MPU6050 mpu_1(0x69); 
MPU6050 mpu_2(0x69);
MPU6050 mpu_3(0x69);
MPU6050 mpu_4(0x69);

int mpu_11 = 4;
int mpu_22 = 5;
int mpu_33 = 6;
int mpu_44 = 7;

#define LED_PIN 13 // (Arduino is 13, Teensy is 11, Teensy++ is 6)



#define _BV(bit) (1 << (bit)) //

bool blinkState = false;

// MPU control/status vars
bool dmpReady_1 = false;  // set true if DMP init was successful
bool dmpReady_2 = false;  // set true if DMP init was successful
bool dmpReady_3 = false;  // set true if DMP init was successful
bool dmpReady_4 = false;  // set true if DMP init was successful
uint8_t mpuIntStatus_1;  // holds actual interrupt status byte from MPU
uint8_t mpuIntStatus_2;  // holds actual interrupt status byte from MPU
uint8_t mpuIntStatus_3;  // holds actual interrupt status byte from MPU
uint8_t mpuIntStatus_4;  // holds actual interrupt status byte from MPU
uint8_t devStatus_1;      // return status after each device operation (0 = success, !0 = error)
uint8_t devStatus_2;      // return status after each device operation (0 = success, !0 = error)
uint8_t devStatus_3;      // return status after each device operation (0 = success, !0 = error)
uint8_t devStatus_4;      // return status after each device operation (0 = success, !0 = error)
uint16_t packetSize_1;    // expected DMP packet size (default is 42 bytes)
uint16_t packetSize_2;    // expected DMP packet size (default is 42 bytes)
uint16_t packetSize_3;    // expected DMP packet size (default is 42 bytes)
uint16_t packetSize_4;    // expected DMP packet size (default is 42 bytes)
uint16_t fifoCount_1;     // count of all bytes currently in FIFO
uint16_t fifoCount_2;     // count of all bytes currently in FIFO
uint16_t fifoCount_3;     // count of all bytes currently in FIFO
uint16_t fifoCount_4;     // count of all bytes currently in FIFO
uint8_t fifoBuffer_1[64]; // FIFO storage buffer
uint8_t fifoBuffer_2[64]; // FIFO storage buffer
uint8_t fifoBuffer_3[64]; // FIFO storage buffer
uint8_t fifoBuffer_4[64]; // FIFO storage buffer

// orientation/motion vars
Quaternion q;           // [w, x, y, z]         quaternion container
VectorInt16 aa;         // [x, y, z]            accel sensor measurements
VectorInt16 aaReal;     // [x, y, z]            gravity-free accel sensor measurements
VectorInt16 aaWorld;    // [x, y, z]            world-frame accel sensor measurements
VectorFloat gravity;    // [x, y, z]            gravity vector
float ypr_1[3];         // [yaw, pitch, roll]   yaw/pitch/roll container and gravity vector
float ypr_2[3];         // [yaw, pitch, roll]   yaw/pitch/roll container and gravity vector
float ypr_3[3];         // [yaw, pitch, roll]   yaw/pitch/roll container and gravity vector
float ypr_4[3];         // [yaw, pitch, roll]   yaw/pitch/roll container and gravity vector

//extra stuff
int global_fifo_count_1 = 0; //made global so can monitor from outside GetIMUHeadingDeg() fcn
int global_fifo_count_2 = 0; //made global so can monitor from outside GetIMUHeadingDeg() fcn
int global_fifo_count_3 = 0; //made global so can monitor from outside GetIMUHeadingDeg() fcn
int global_fifo_count_4 = 0; //made global so can monitor from outside GetIMUHeadingDeg() fcn

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

float x1,x2,x3,x4,y1,y2,y3,y4,z1,z2,z3,z4;


// ================================================================
// ===                      INITIAL SETUP                       ===
// ================================================================


void setup() {
  // Wait for Serial to connect

     #if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
  Wire.begin();
  Wire.setClock(400000); // 400kHz I2C clock. Comment this line if having compilation difficulties
#elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
  Fastwire::setup(400, true);
#endif
    pinMode(mpu_11,OUTPUT);
    pinMode(mpu_22,OUTPUT);
    pinMode(mpu_33,OUTPUT);
    pinMode(mpu_44,OUTPUT);

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

///////////////////////////////////////////////////////////////////// initialize device
  Serial.println(F("Initializing MPU6050..."));
  digitalWrite(mpu_11,HIGH);
  mpu_1.initialize();
  // verify connection
  Serial.println(F("Testing device connections..."));
  Serial.println(mpu_1.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));
  // load and configure the DMP
  Serial.println(F("Initializing DMP..."));
  devStatus_1 = mpu_1.dmpInitialize();
  // supply your own gyro offsets here, scaled for min sensitivity
  mpu_1.setXGyroOffset(51);
  mpu_1.setYGyroOffset(8);
  mpu_1.setZGyroOffset(21);
  mpu_1.setXAccelOffset(1150);
  mpu_1.setYAccelOffset(-50);
  mpu_1.setZAccelOffset(1060);
  // make sure it worked (returns 0 if so)
  if (devStatus_1 == 0)
  {
    // Calibration Time: generate offsets and calibrate our MPU6050
    mpu_1.CalibrateAccel(6);
    mpu_1.CalibrateGyro(6);
    mpu_1.PrintActiveOffsets();
    // turn on the DMP, now that it's ready
    Serial.println(F("Enabling DMP..."));
    mpu_1.setDMPEnabled(true);
    // set our DMP Ready flag so the main loop() function knows it's okay to use it
    Serial.println(F("DMP ready! Waiting for first interrupt..."));
    dmpReady_1 = true;
    // get expected DMP packet size for later comparison
    packetSize_1 = mpu_1.dmpGetFIFOPacketSize();
  }
  else
  {
    // ERROR!
    // 1 = initial memory load failed
    // 2 = DMP configuration updates failed
    // (if it's going to break, usually the code will be 1)
    Serial.print(F("DMP Initialization failed (code "));
    Serial.print(devStatus_1);
    Serial.println(F(")"));
  }
  // configure LED for output
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(mpu_11,LOW);

////////////////////////////////////////////////////////////////////////////

  Serial.println(F("Initializing MPU6050..."));
  digitalWrite(mpu_22,HIGH);
  mpu_2.initialize();
  // verify connection
  Serial.println(F("Testing device connections..."));
  Serial.println(mpu_2.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));
  // load and configure the DMP
  Serial.println(F("Initializing DMP..."));
  devStatus_2 = mpu_2.dmpInitialize();
  // supply your own gyro offsets here, scaled for min sensitivity
  mpu_2.setXGyroOffset(51);
  mpu_2.setYGyroOffset(8);
  mpu_2.setZGyroOffset(21);
  mpu_2.setXAccelOffset(1150);
  mpu_2.setYAccelOffset(-50);
  mpu_2.setZAccelOffset(1060);
  // make sure it worked (returns 0 if so)
  if (devStatus_2 == 0)
  {
    // Calibration Time: generate offsets and calibrate our MPU6050
    mpu_2.CalibrateAccel(6);
    mpu_2.CalibrateGyro(6);
    mpu_2.PrintActiveOffsets();
    // turn on the DMP, now that it's ready
    Serial.println(F("Enabling DMP..."));
    mpu_2.setDMPEnabled(true);
    // set our DMP Ready flag so the main loop() function knows it's okay to use it
    Serial.println(F("DMP ready! Waiting for first interrupt..."));
    dmpReady_2 = true;
    // get expected DMP packet size for later comparison
    packetSize_2 = mpu_2.dmpGetFIFOPacketSize();
  }
  else
  {
    // ERROR!
    // 1 = initial memory load failed
    // 2 = DMP configuration updates failed
    // (if it's going to break, usually the code will be 1)
    Serial.print(F("DMP Initialization failed (code "));
    Serial.print(devStatus_2);
    Serial.println(F(")"));
  }

  // configure LED for output
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(mpu_22,LOW);

////////////////////////////////////////////////////////////////////////////

  Serial.println(F("Initializing MPU6050..."));
  digitalWrite(mpu_22,HIGH);
  mpu_2.initialize();
  // verify connection
  Serial.println(F("Testing device connections..."));
  Serial.println(mpu_2.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));
  // load and configure the DMP
  Serial.println(F("Initializing DMP..."));
  devStatus_2 = mpu_2.dmpInitialize();
  // supply your own gyro offsets here, scaled for min sensitivity
  mpu_2.setXGyroOffset(51);
  mpu_2.setYGyroOffset(8);
  mpu_2.setZGyroOffset(21);
  mpu_2.setXAccelOffset(1150);
  mpu_2.setYAccelOffset(-50);
  mpu_2.setZAccelOffset(1060);
  // make sure it worked (returns 0 if so)
  if (devStatus_2 == 0)
  {
    // Calibration Time: generate offsets and calibrate our MPU6050
    mpu_2.CalibrateAccel(6);
    mpu_2.CalibrateGyro(6);
    mpu_2.PrintActiveOffsets();
    // turn on the DMP, now that it's ready
    Serial.println(F("Enabling DMP..."));
    mpu_2.setDMPEnabled(true);
    // set our DMP Ready flag so the main loop() function knows it's okay to use it
    Serial.println(F("DMP ready! Waiting for first interrupt..."));
    dmpReady_2 = true;
    // get expected DMP packet size for later comparison
    packetSize_2 = mpu_2.dmpGetFIFOPacketSize();
  }
  else
  {
    // ERROR!
    // 1 = initial memory load failed
    // 2 = DMP configuration updates failed
    // (if it's going to break, usually the code will be 1)
    Serial.print(F("DMP Initialization failed (code "));
    Serial.print(devStatus_2);
    Serial.println(F(")"));
  }

  // configure LED for output
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(mpu_22,LOW);////////////////////////////////////////////////////////////////////////////

  Serial.println(F("Initializing MPU6050..."));
  digitalWrite(mpu_33,HIGH);
  mpu_3.initialize();
  // verify connection
  Serial.println(F("Testing device connections..."));
  Serial.println(mpu_3.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));
  // load and configure the DMP
  Serial.println(F("Initializing DMP..."));
  devStatus_3 = mpu_3.dmpInitialize();
  // supply your own gyro offsets here, scaled for min sensitivity
  mpu_3.setXGyroOffset(51);
  mpu_3.setYGyroOffset(8);
  mpu_3.setZGyroOffset(21);
  mpu_3.setXAccelOffset(1150);
  mpu_3.setYAccelOffset(-50);
  mpu_3.setZAccelOffset(1060);
  // make sure it worked (returns 0 if so)
  if (devStatus_3 == 0)
  {
    // Calibration Time: generate offsets and calibrate our MPU6050
    mpu_3.CalibrateAccel(6);
    mpu_3.CalibrateGyro(6);
    mpu_3.PrintActiveOffsets();
    // turn on the DMP, now that it's ready
    Serial.println(F("Enabling DMP..."));
    mpu_3.setDMPEnabled(true);
    // set our DMP Ready flag so the main loop() function knows it's okay to use it
    Serial.println(F("DMP ready! Waiting for first interrupt..."));
    dmpReady_3 = true;
    // get expected DMP packet size for later comparison
    packetSize_3 = mpu_3.dmpGetFIFOPacketSize();
  }
  else
  {
    // ERROR!
    // 1 = initial memory load failed
    // 2 = DMP configuration updates failed
    // (if it's going to break, usually the code will be 1)
    Serial.print(F("DMP Initialization failed (code "));
    Serial.print(devStatus_3);
    Serial.println(F(")"));
  }

  // configure LED for output
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(mpu_33,LOW);

  ////////////////////////////////////////////////////////////////////////////

  Serial.println(F("Initializing MPU6050..."));
  digitalWrite(mpu_44,HIGH);
  mpu_4.initialize();
  // verify connection
  Serial.println(F("Testing device connections..."));
  Serial.println(mpu_4.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));
  // load and configure the DMP
  Serial.println(F("Initializing DMP..."));
  devStatus_4 = mpu_4.dmpInitialize();
  // supply your own gyro offsets here, scaled for min sensitivity
  mpu_4.setXGyroOffset(51);
  mpu_4.setYGyroOffset(8);
  mpu_4.setZGyroOffset(21);
  mpu_4.setXAccelOffset(1150);
  mpu_4.setYAccelOffset(-50);
  mpu_4.setZAccelOffset(1060);
  // make sure it worked (returns 0 if so)
  if (devStatus_4 == 0)
  {
    // Calibration Time: generate offsets and calibrate our MPU6050
    mpu_4.CalibrateAccel(6);
    mpu_4.CalibrateGyro(6);
    mpu_4.PrintActiveOffsets();
    // turn on the DMP, now that it's ready
    Serial.println(F("Enabling DMP..."));
    mpu_4.setDMPEnabled(true);
    // set our DMP Ready flag so the main loop() function knows it's okay to use it
    Serial.println(F("DMP ready! Waiting for first interrupt..."));
    dmpReady_4 = true;
    // get expected DMP packet size for later comparison
    packetSize_4 = mpu_4.dmpGetFIFOPacketSize();
  }
  else
  {
    // ERROR!
    // 1 = initial memory load failed
    // 2 = DMP configuration updates failed
    // (if it's going to break, usually the code will be 1)
    Serial.print(F("DMP Initialization failed (code "));
    Serial.print(devStatus_4);
    Serial.println(F(")"));
  }

  // configure LED for output
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(mpu_44,LOW);
}




void loop() {
  float duration = millis();

  if (!dmpReady_1) {
    return;
  }

  digitalWrite(mpu_11,HIGH);
  if (mpu_1.dmpPacketAvailable())
  {
    prev=millis();
    GetIMUHeadingDeg_1(); //retreive the most current yaw value from IMU
    blinkState = !blinkState;
    digitalWrite(LED_PIN, blinkState);
  }
  digitalWrite(mpu_11,LOW);


  if (!dmpReady_2) {
    return;
  }

  digitalWrite(mpu_22,HIGH);
  if (mpu_2.dmpPacketAvailable())
  {
    GetIMUHeadingDeg_2(); //retreive the most current yaw value from IMU
    blinkState = !blinkState;
    digitalWrite(LED_PIN, blinkState);
  }
  digitalWrite(mpu_22,LOW);


  digitalWrite(mpu_33,HIGH);
  if (mpu_3.dmpPacketAvailable())
  {
    GetIMUHeadingDeg_3(); //retreive the most current yaw value from IMU
    blinkState = !blinkState;
    digitalWrite(LED_PIN, blinkState);
  }
  digitalWrite(mpu_33,LOW);

  digitalWrite(mpu_44,HIGH);
  if (mpu_4.dmpPacketAvailable())
  {
    GetIMUHeadingDeg_4(); //retreive the most current yaw value from IMU
    blinkState = !blinkState;
    digitalWrite(LED_PIN, blinkState);
    Serial.print("\t");
    Serial.println(millis()-prev);
  }
  digitalWrite(mpu_44,LOW);

  // Calculate x value to feed to the model
  float x_val = 1;


  // Copy value to input buffer (tensor)
  model_input->data.f[0] = x1;
  model_input->data.f[1] = y1;
  model_input->data.f[2] = z1;
  model_input->data.f[3] = x2;
  model_input->data.f[4] = y2;
  model_input->data.f[5] = z2;
  model_input->data.f[6] = x3;
  model_input->data.f[7] = y3;
  model_input->data.f[8] = z3;
  model_input->data.f[9] = x4;
  model_input->data.f[10] = y4;
  model_input->data.f[11] = z4;

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



void GetIMUHeadingDeg_1()
{
  // At least one data packet is available

  mpuIntStatus_1 = mpu_1.getIntStatus();  
  fifoCount_1 = mpu_1.getFIFOCount();// get current FIFO count

  // check for overflow (this should never happen unless our code is too inefficient)
  if ((mpuIntStatus_1 & _BV(MPU6050_INTERRUPT_FIFO_OFLOW_BIT)) || fifoCount_1 >= 1024)
  {
    // reset so we can continue cleanly
    mpu_1.resetFIFO();
    Serial.println(F("FIFO overflow!"));

    // otherwise, check for DMP data ready interrupt (this should happen frequently)
  }
  else if (mpuIntStatus_1 & _BV(MPU6050_INTERRUPT_DMP_INT_BIT))
  {
    // read all available packets from FIFO
    while (fifoCount_1 >= packetSize_1) // Lets catch up to NOW, in case someone is using the dreaded delay()!
    {
      mpu_1.getFIFOBytes(fifoBuffer_1, packetSize_1);
      // track FIFO count here in case there is > 1 packet available
      // (this lets us immediately read more without waiting for an interrupt)
      fifoCount_1 -= packetSize_1;
    }
    global_fifo_count_1 = mpu_1.getFIFOCount(); //should be zero here

    // display Euler angles in degrees
    mpu_1.dmpGetQuaternion(&q, fifoBuffer_1);
    mpu_1.dmpGetGravity(&gravity, &q);
    mpu_1.dmpGetYawPitchRoll(ypr_1, &q, &gravity);
  }

    x1 = (ypr_1[0] * 180 / M_PI);
    y1 = (ypr_1[1] * 180 / M_PI);
    z1 = (ypr_1[2] * 180 / M_PI);
}



void GetIMUHeadingDeg_2()
{
  // At least one data packet is available

  mpuIntStatus_2 = mpu_2.getIntStatus();  
  fifoCount_2 = mpu_2.getFIFOCount();// get current FIFO count

  // check for overflow (this should never happen unless our code is too inefficient)
  if ((mpuIntStatus_2 & _BV(MPU6050_INTERRUPT_FIFO_OFLOW_BIT)) || fifoCount_2 >= 1024)
  {
    // reset so we can continue cleanly
    mpu_2.resetFIFO();
    Serial.println(F("FIFO overflow!"));

    // otherwise, check for DMP data ready interrupt (this should happen frequently)
  }
  else if (mpuIntStatus_2 & _BV(MPU6050_INTERRUPT_DMP_INT_BIT))
  {
    // read all available packets from FIFO
    while (fifoCount_2 >= packetSize_2) // Lets catch up to NOW, in case someone is using the dreaded delay()!
    {
      mpu_2.getFIFOBytes(fifoBuffer_2, packetSize_2);
      // track FIFO count here in case there is > 1 packet available
      // (this lets us immediately read more without waiting for an interrupt)
      fifoCount_2 -= packetSize_2;
    }
    global_fifo_count_2 = mpu_2.getFIFOCount(); //should be zero here

    // display Euler angles in degrees
    mpu_2.dmpGetQuaternion(&q, fifoBuffer_2);
    mpu_2.dmpGetGravity(&gravity, &q);
    mpu_2.dmpGetYawPitchRoll(ypr_2, &q, &gravity);
  }

    x2 = (ypr_1[0] * 180 / M_PI);
    y2 = (ypr_1[1] * 180 / M_PI);
    z2 = (ypr_1[2] * 180 / M_PI);
}


void GetIMUHeadingDeg_3()
{
  // At least one data packet is available

  mpuIntStatus_3 = mpu_3.getIntStatus();  
  fifoCount_3 = mpu_3.getFIFOCount();// get current FIFO count

  // check for overflow (this should never happen unless our code is too inefficient)
  if ((mpuIntStatus_3 & _BV(MPU6050_INTERRUPT_FIFO_OFLOW_BIT)) || fifoCount_3 >= 1024)
  {
    // reset so we can continue cleanly
    mpu_3.resetFIFO();
    Serial.println(F("FIFO overflow!"));

    // otherwise, check for DMP data ready interrupt (this should happen frequently)
  }
  else if (mpuIntStatus_3 & _BV(MPU6050_INTERRUPT_DMP_INT_BIT))
  {
    // read all available packets from FIFO
    while (fifoCount_3 >= packetSize_3) // Lets catch up to NOW, in case someone is using the dreaded delay()!
    {
      mpu_3.getFIFOBytes(fifoBuffer_3, packetSize_3);
      // track FIFO count here in case there is > 1 packet available
      // (this lets us immediately read more without waiting for an interrupt)
      fifoCount_3 -= packetSize_3;
    }
    global_fifo_count_3 = mpu_3.getFIFOCount(); //should be zero here

    // display Euler angles in degrees
    mpu_3.dmpGetQuaternion(&q, fifoBuffer_3);
    mpu_3.dmpGetGravity(&gravity, &q);
    mpu_3.dmpGetYawPitchRoll(ypr_3, &q, &gravity);
  }

    x3 = (ypr_1[0] * 180 / M_PI);
    y3 = (ypr_1[1] * 180 / M_PI);
    z3 = (ypr_1[2] * 180 / M_PI);
}


void GetIMUHeadingDeg_4()
{
  // At least one data packet is available

  mpuIntStatus_4 = mpu_4.getIntStatus();  
  fifoCount_4 = mpu_4.getFIFOCount();// get current FIFO count

  // check for overflow (this should never happen unless our code is too inefficient)
  if ((mpuIntStatus_4 & _BV(MPU6050_INTERRUPT_FIFO_OFLOW_BIT)) || fifoCount_4 >= 1024)
  {
    // reset so we can continue cleanly
    mpu_4.resetFIFO();
    Serial.println(F("FIFO overflow!"));

    // otherwise, check for DMP data ready interrupt (this should happen frequently)
  }
  else if (mpuIntStatus_4 & _BV(MPU6050_INTERRUPT_DMP_INT_BIT))
  {
    // read all available packets from FIFO
    while (fifoCount_4 >= packetSize_4) // Lets catch up to NOW, in case someone is using the dreaded delay()!
    {
      mpu_4.getFIFOBytes(fifoBuffer_4, packetSize_4);
      // track FIFO count here in case there is > 1 packet available
      // (this lets us immediately read more without waiting for an interrupt)
      fifoCount_4 -= packetSize_4;
    }
    global_fifo_count_4 = mpu_4.getFIFOCount(); //should be zero here

    // display Euler angles in degrees
    mpu_4.dmpGetQuaternion(&q, fifoBuffer_4);
    mpu_4.dmpGetGravity(&gravity, &q);
    mpu_4.dmpGetYawPitchRoll(ypr_4, &q, &gravity);
  }

    x4 = (ypr_1[0] * 180 / M_PI);
    y4 = (ypr_1[1] * 180 / M_PI);
    z4 = (ypr_1[2] * 180 / M_PI);
}
