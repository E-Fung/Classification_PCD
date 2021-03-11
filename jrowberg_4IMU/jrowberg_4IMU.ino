/*
    Name:       Arduino MPU6050 Polling Test.ino
    Created:  10/5/2019 1:31:08 PM
    Author:     FRANKWIN10\Frank

  This is a complete, working MPU6050 (GY-521) example using polling vs interrupts.
  It is based on Jeff Rowberg's MPU6050_DMP6_using_DMP_V6.12 example.  After confirming
  that the example worked properly using interrupts, I modified it to remove the need
  for the interrupt line.  
*/
#include "I2Cdev.h"
#include "MPU6050V6.h"  
#include<stdlib.h>




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

// ================================================================
// ===                      INITIAL SETUP                       ===
// ================================================================

void setup()
{

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
    
    // General Setup

  Serial.begin(115200);
  while (!Serial); // wait for Leonardo enumeration, others continue immediately

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

// ================================================================
// ===                    MAIN PROGRAM LOOP                     ===
// ================================================================
void loop()
{
  float prev;
  
  // if programming failed, don't try to do anything
  
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

  //other program stuff block - executes every IMU_CHECK_INTERVAL_MSEC Msec
  //for this test program, there's nothing here except diagnostics printouts
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

    Serial.print("ypr\t");
    Serial.print(ypr_1[0] * 180 / M_PI);
    Serial.print("\t");
    Serial.print(ypr_1[1] * 180 / M_PI);
    Serial.print("\t");
    Serial.print(ypr_1[2] * 180 / M_PI);
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

    Serial.print(" ypr\t");
    Serial.print(ypr_2[0] * 180 / M_PI);
    Serial.print("\t");
    Serial.print(ypr_2[1] * 180 / M_PI);
    Serial.print("\t");
    Serial.print(ypr_2[2] * 180 / M_PI);
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

    Serial.print(" ypr\t");
    Serial.print(ypr_3[0] * 180 / M_PI);
    Serial.print("\t");
    Serial.print(ypr_3[1] * 180 / M_PI);
    Serial.print("\t");
    Serial.print(ypr_3[2] * 180 / M_PI);
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

    Serial.print(" ypr\t");
    Serial.print(ypr_4[0] * 180 / M_PI);
    Serial.print("\t");
    Serial.print(ypr_4[1] * 180 / M_PI);
    Serial.print("\t");
    Serial.print(ypr_4[2] * 180 / M_PI);
}
