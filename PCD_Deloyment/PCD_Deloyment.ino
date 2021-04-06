#include <cQueue.h>
#include <Arduino.h>
#include <TensorFlowLite.h>
#include <stdlib.h>
#include "I2Cdev.h"
#include "MPU6050V6.h"
#include "PCD_Model.h"

// Import TensorFlow stuff
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define _BV(bit) (1 << (bit))
#define M_PI 3.141592653589793238462643
#define LED_PIN 13 // (Arduino is 13, Teensy is 11, Teensy++ is 6)

// ================================================================
// ===                      DEBUGGING                           ===
// ================================================================

#define DEBUG 0
#define COUNTER_DEBUG
//#define ANGLE_DEBUG
//#define MODEL_DEBUG
//#define SOFTMAX_DEBUG

// ================================================================
// ===                      INITIALIZING                        ===
// ================================================================

Queue_t myqueue; //Queue
uint8_t in = 0;

int myCounter[6] = {0, 0, 0, 0, 0, 0}; //Counter
float angles[12]; //Angles

int queue_size = 50;
int time_threshold = queue_size * 0.5; //Thresholds
float myThresholds[5] = {0.7, 0.7, 0.7, 0.7, 0.7};

int LED_01;
int LED_02;
int LED_03;
int LED_04;
int LED_05;
int LED_06;
int last_pos = 10;
MPU6050 mpu_1(0x69); //IMUs
MPU6050 mpu_2(0x69);
MPU6050 mpu_3(0x69);
MPU6050 mpu_4(0x69);
int mpu_11 = 4;
int mpu_22 = 5;
int mpu_33 = 6;
int mpu_44 = 7;
bool blinkState = false; // MPU control/status vars
bool dmpReady_1 = false; // set true if DMP init was successful
bool dmpReady_2 = false; // set true if DMP init was successful
bool dmpReady_3 = false; // set true if DMP init was successful
bool dmpReady_4 = false; // set true if DMP init was successful
uint8_t mpuIntStatus;    // holds actual interrupt status byte from MPU
uint8_t devStatus_1;     // return status after each device operation (0 = success, !0 = error)
uint8_t devStatus_2;     // return status after each device operation (0 = success, !0 = error)
uint8_t devStatus_3;     // return status after each device operation (0 = success, !0 = error)
uint8_t devStatus_4;     // return status after each device operation (0 = success, !0 = error)
uint16_t packetSize_1;   // expected DMP packet size (default is 42 bytes)
uint16_t packetSize_2;   // expected DMP packet size (default is 42 bytes)
uint16_t packetSize_3;   // expected DMP packet size (default is 42 bytes)
uint16_t packetSize_4;   // expected DMP packet size (default is 42 bytes)
uint16_t fifoCount;      // count of all bytes currently in FIFO
uint16_t fifoCount_2;    // count of all bytes currently in FIFO
uint16_t fifoCount_3;    // count of all bytes currently in FIFO
uint16_t fifoCount_4;    // count of all bytes currently in FIFO
uint8_t fifoBuffer[64];  // FIFO storage buffer
Quaternion q;        // [w, x, y, z]         quaternion container
VectorInt16 aa;      // [x, y, z]            accel sensor measurements
VectorInt16 aaReal;  // [x, y, z]            gravity-free accel sensor measurements
VectorInt16 aaWorld; // [x, y, z]            world-frame accel sensor measurements
VectorFloat gravity; // [x, y, z]            gravity vector
float ypr[3];        // [yaw, pitch, roll]   yaw/pitch/roll container and gravity vector
float *ypr_ptr = ypr;
int global_fifo_count_1 = 0; //made global so can monitor from outside GetIMUHeadingDeg() fcn
int global_fifo_count_2 = 0; //made global so can monitor from outside GetIMUHeadingDeg() fcn
int global_fifo_count_3 = 0; //made global so can monitor from outside GetIMUHeadingDeg() fcn
int global_fifo_count_4 = 0; //made global so can monitor from outside GetIMUHeadingDeg() fcn

// ================================================================
// ===                   TENSORFLOW SETUP                       ===
// ================================================================

namespace
{
tflite::ErrorReporter *error_reporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *model_input = nullptr;
TfLiteTensor *model_output = nullptr;

// Create an area of memory to use for input, output, and other TensorFlow
// arrays. You'll need to adjust this by combiling, running, and looking
// for errors.
constexpr int kTensorArenaSize = 6 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}

// ================================================================
// ===                      INITIAL SETUP                       ===
// ================================================================

void setup()
{
  delay(1000);
  Serial.println("Starting");// Wait for Serial to connect
  pinMode(mpu_11, OUTPUT);
  pinMode(mpu_22, OUTPUT);
  pinMode(mpu_33, OUTPUT);
  pinMode(mpu_44, OUTPUT);

  pinMode(LED_01, OUTPUT);
  pinMode(LED_02, OUTPUT);
  pinMode(LED_03, OUTPUT);
  pinMode(LED_04, OUTPUT);
  pinMode(LED_05, OUTPUT);
  pinMode(LED_06, OUTPUT);

  digitalWrite(LED_01, LOW);
  digitalWrite(LED_02, LOW);
  digitalWrite(LED_03, LOW);
  digitalWrite(LED_04, LOW);
  digitalWrite(LED_05, LOW);
  digitalWrite(LED_06, LOW);

  //////////////////////////////////////////////////////////////////////////// Serial

#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
  Wire.begin();
  Wire.setClock(400000); // 400kHz I2C clock. Comment this line if having compilation difficulties
#elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
  Fastwire::setup(400, true);
#endif
#if DEBUG
  while (!Serial)
    ;
#endif

  //////////////////////////////////////////////////////////////////////////// TensorFlow

  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  model = tflite::GetModel(PCD_Model);  // Map the model into a usable data structure
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    error_reporter->Report("Model version does not match Schema");
    while (1)
      ;
  }
  // Pull in only needed operations (should match NN layers)
  // Available ops:
  //  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/micro_ops.h
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    while (1)
      ;
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
  Serial.print("Dim 3 size: ");
  Serial.println(model_input->dims->data[2]);
  Serial.print("Input type: ");
  Serial.println(model_input->type);
#endif

  //////////////////////////////////////////////////////////////////////////// IMU 1

  Serial.println(F("Initializing MPU6050..."));
  digitalWrite(mpu_11, HIGH);
  mpu_1.initialize();
  // verify connection
  Serial.println(F("Testing device connections..."));
  Serial.println(mpu_1.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));
  // load and configure the DMP
  Serial.println(F("Initializing DMP..."));
  devStatus_1 = mpu_1.dmpInitialize();
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
    Serial.print(F("DMP 1 Initialization failed (code "));
    Serial.print(devStatus_1);
    Serial.println(F(")"));
  }
  // configure LED for output
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(mpu_11, LOW);

  //////////////////////////////////////////////////////////////////////////// IMU 2

  Serial.println(F("Initializing MPU6050..."));
  digitalWrite(mpu_22, HIGH);
  mpu_2.initialize();
  // verify connection
  Serial.println(F("Testing device connections..."));
  Serial.println(mpu_2.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));
  // load and configure the DMP
  Serial.println(F("Initializing DMP..."));
  devStatus_2 = mpu_2.dmpInitialize();
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
    Serial.print(F("DMP 2 Initialization failed (code "));
    Serial.print(devStatus_2);
    Serial.println(F(")"));
  }
  // configure LED for output
  digitalWrite(mpu_22, LOW);

  //////////////////////////////////////////////////////////////////////////// IMU 3

  Serial.println(F("Initializing MPU6050..."));
  digitalWrite(mpu_33, HIGH);
  mpu_3.initialize();
  // verify connection
  Serial.println(F("Testing device connections..."));
  Serial.println(mpu_3.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));
  // load and configure the DMP
  Serial.println(F("Initializing DMP..."));
  devStatus_3 = mpu_3.dmpInitialize();
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
    Serial.println(F("DMP 3 ready! Waiting for first interrupt..."));
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
  digitalWrite(mpu_33, LOW);

  //////////////////////////////////////////////////////////////////////////// IMU 4

  Serial.println(F("Initializing MPU6050..."));
  digitalWrite(mpu_44, HIGH);
  mpu_4.initialize();
  // verify connection
  Serial.println(F("Testing device connections..."));
  Serial.println(mpu_4.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));
  // load and configure the DMP
  Serial.println(F("Initializing DMP..."));
  devStatus_4 = mpu_4.dmpInitialize();
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
    Serial.print(F("DMP 4 Initialization failed (code "));
    Serial.print(devStatus_4);
    Serial.println(F(")"));
  }
  // configure LED for output
  digitalWrite(mpu_44, LOW);

  //////////////////////////////////////////////////////////////////////////// QUEUE

  q_init(&myqueue, sizeof(in), queue_size, FIFO, true);

}

// ================================================================
// ===                      SOFTMAX FUNCTION                    ===
// ================================================================
//This function converts the model outputs through a softmax filter and prints the results
void convert_softmax(float y_val[]) {
  float denom = 0;
  for (int i = 0; i < 5; i++) {
    denom += exp(y_val[i]);
  }
  for (int i = 0; i < 5; i++) {
    y_val[i] = (exp(y_val[i])) / denom;
  }
}

// ================================================================
// ===                PRINT SOFTMAX FUNCTION                    ===
// ================================================================
void print_y_val(float y_val[]) {
  Serial.print("Softmax:");
  for (int i = 0; i < 5; i++) {
    Serial.print("\t");
    Serial.print(y_val[i]);
  }
}


// ================================================================
// ===                      UPDATE IMU FUNCTION                 ===
// ================================================================
void GetIMUHeadingDeg(MPU6050 *curr_mpu, uint16_t packetSize, int *global_fifo_count)
{
  // At least one data packet is available
  mpuIntStatus = curr_mpu->getIntStatus();
  fifoCount = curr_mpu->getFIFOCount(); // get current FIFO count
  // check for overflow (this should never happen unless our code is too inefficient)
  if ((mpuIntStatus & _BV(MPU6050_INTERRUPT_FIFO_OFLOW_BIT)) || fifoCount >= 1024)
  {
    // reset so we can continue cleanly
    curr_mpu->resetFIFO();
    Serial.println(F("FIFO overflow!"));
    // otherwise, check for DMP data ready interrupt (this should happen frequently)
  }
  else if (mpuIntStatus & _BV(MPU6050_INTERRUPT_DMP_INT_BIT))
  {
    // read all available packets from FIFO
    while (fifoCount >= packetSize) // Lets catch up to NOW, in case someone is using the dreaded delay()!
    {
      curr_mpu->getFIFOBytes(fifoBuffer, packetSize);
      // track FIFO count here in case there is > 1 packet available
      // (this lets us immediately read more without waiting for an interrupt)
      fifoCount -= packetSize;
    }
    *global_fifo_count = curr_mpu->getFIFOCount(); //should be zero here
    // display Euler angles in degrees
    curr_mpu->dmpGetQuaternion(&q, fifoBuffer);
    curr_mpu->dmpGetGravity(&gravity, &q);
    curr_mpu->dmpGetYawPitchRoll(ypr, &q, &gravity);
  }
  ypr[0] = (ypr[0] * 180 / M_PI);
  ypr[1] = (ypr[1] * 180 / M_PI);
  ypr[2] = (ypr[2] * 180 / M_PI);
}


// ================================================================
// ===                      MAIN LOOP                           ===
// ================================================================
void loop()
{
  float duration = millis();
  //////////////////////////////////////////////////////////////////////////// IMU 1
  if (!dmpReady_1)
  {
    return;
  }
  digitalWrite(mpu_11, HIGH);
  if (mpu_1.dmpPacketAvailable())
  {
    GetIMUHeadingDeg(&mpu_1, packetSize_1, &global_fifo_count_1); //retreive the most current yaw
    angles[0] = ypr[0] / 180;
    angles[1] = ypr[1] / 180;
    angles[2] = ypr[2] / 180;
    model_input->data.f[0] = angles[0];
    model_input->data.f[1] = angles[1];
    model_input->data.f[2] = angles[2];
  }
  digitalWrite(mpu_11, LOW);

  //////////////////////////////////////////////////////////////////////////// IMU 2
  if (!dmpReady_2)
  {
    return;
  }
  digitalWrite(mpu_22, HIGH);
  if (mpu_2.dmpPacketAvailable())
  {
    GetIMUHeadingDeg(&mpu_2, packetSize_2, &global_fifo_count_2); //retreive the most current yaw
    angles[3] = ypr[0] / 180;
    angles[4] = ypr[1] / 180;
    angles[5] = ypr[2] / 180;
    model_input->data.f[3] = angles[3];
    model_input->data.f[4] = angles[4];
    model_input->data.f[5] = angles[5];
  }
  digitalWrite(mpu_22, LOW);

  //////////////////////////////////////////////////////////////////////////// IMU 3
  if (!dmpReady_3)
  {
    return;
  }
  digitalWrite(mpu_33, HIGH);
  if (mpu_3.dmpPacketAvailable())
  {
    GetIMUHeadingDeg(&mpu_3, packetSize_3, &global_fifo_count_3); //retreive the most current yaw
    angles[6] = ypr[0] / 180;
    angles[7] = ypr[1] / 180;
    angles[8] = ypr[2] / 180;
    model_input->data.f[6] = angles[6];
    model_input->data.f[7] = angles[7];
    model_input->data.f[8] = angles[8];
  }
  digitalWrite(mpu_33, LOW);

  //////////////////////////////////////////////////////////////////////////// IMU 4
  if (!dmpReady_4)
  {
    return;
  }
  digitalWrite(mpu_44, HIGH);
  if (mpu_4.dmpPacketAvailable())
  {
    GetIMUHeadingDeg(&mpu_4, packetSize_4, &global_fifo_count_4); //retreive the most current yaw
    angles[9] = ypr[0] / 180;
    angles[10] = ypr[1] / 180;
    angles[11] = ypr[2] / 180;
    model_input->data.f[9] = angles[9];
    model_input->data.f[10] = angles[10];
    model_input->data.f[11] = angles[11];
  }
  digitalWrite(mpu_44, LOW);
  //////////////////////////////////////////////////////////////////////////// MODEL
  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk)
  {
    error_reporter->Report("Invoke failed on input: ");
  }
  // Read predicted y value from output buffer (tensor)
  float y_val[5];
  y_val[0] = model_output->data.f[0];
  y_val[1] = model_output->data.f[1];
  y_val[2] = model_output->data.f[2];
  y_val[3] = model_output->data.f[3];
  y_val[4] = model_output->data.f[4];
  //////////////////////////////////////////////////////////////////////////// VALUE PRINTING
#ifdef ANGLE_DEBUG
  for (int i = 0; i < 12; i++) {
    Serial.print(angles[i]);
    Serial.print("\t");
  }
#endif

#ifdef MODEL_DEBUG
  Serial.print("ML Model Output: ");
  Serial.print("\t");
  for (int i = 0; i < 5; i++) {
    Serial.print(y_val[i]);
    Serial.print("\t");
  }
#endif

#ifdef SOFTMAX_DEBUG
  convert_softmax(y_val);
  print_y_val(y_val);
#endif


  int curr_pos = 10;
  int temp = NULL;
  for (int per_itr = 0; per_itr <= 5; per_itr++) {
    if (y_val[per_itr] >= myThresholds[per_itr]) {
      curr_pos = per_itr;
    }
  }
  if (curr_pos == 10) {
    curr_pos = 5;
  }
  if (q_isFull(&myqueue)) {
    q_pop(&myqueue, &temp);
    myCounter[temp]--;
    q_push(&myqueue, &curr_pos);
  }
  else {
    q_push(&myqueue, &curr_pos);
  }
  myCounter[curr_pos]++;

#ifdef COUNTER_DEBUG
  Serial.print("Counter:\t");
  for (int prn_itr = 0; prn_itr < 6; prn_itr++) {
    Serial.print(myCounter[prn_itr]);
    Serial.print("\t");
  }
#endif
  for (int prn_itr = 0; prn_itr < 6; prn_itr++) {
    if (myCounter[prn_itr] > time_threshold) {

      Serial.print("POSTURE:\t");
      Serial.print(prn_itr+1);
      Serial.print("\t");
      if (last_pos != prn_itr && last_pos != 10) {
        digitalWrite(last_pos + 1, LOW);
      }
      switch (prn_itr) {
        case 0:
          digitalWrite(LED_01, HIGH);
          break;
        case 1:
          digitalWrite(LED_02, HIGH);
          break;
        case 2:
          digitalWrite(LED_03, HIGH);
          break;
        case 3:
          digitalWrite(LED_04, HIGH);
          break;
        case 4:
          digitalWrite(LED_05, HIGH);
          break;
        case 5:
          digitalWrite(LED_06, HIGH);
          break;
      }
      last_pos = prn_itr;
    }
  }
  Serial.print("Dur: ");
  Serial.println(millis() - duration);
}
