#include <Wire.h>
#include <MPU6050_light.h>

MPU6050 mpu(Wire);

void setup() { 
  Serial.begin(115200);

  Wire.begin();   // Pro Micro: SDA=2, SCL=3

  Serial.println("Initializing MPU6050...");

  byte status = mpu.begin();   // auto-detect 0x68 or 0x69
  Serial.print("MPU6050 status: ");
  Serial.println(status);

  delay(1000);
}

void loop() {
  mpu.update();

  // Output accelerometer + gyro (6 values)
  Serial.print(mpu.getAccX()); Serial.print(",");
  Serial.print(mpu.getAccY()); Serial.print(",");
  Serial.print(mpu.getAccZ()); Serial.print(",");
  Serial.print(mpu.getGyroX()); Serial.print(",");
  Serial.print(mpu.getGyroY()); Serial.print(",");
  Serial.println(mpu.getGyroZ());

  delay(1000);  // ~20 Hz
}
