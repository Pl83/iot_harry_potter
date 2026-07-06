/*
 * Based on I2C device class (I2Cdev) Arduino sketch for MPU6050 class by Jeff Rowberg <jeff@rowberg.net>
 * and Edge Impulse Data Forwarder Exampe (Arduino) - https://docs.edgeimpulse.com/docs/cli-data-forwarder
 * 
 * Developed by M.Rovai @11May23
 */

#include "MPU6050.h"
#include "Wire.h"

#define FREQUENCY_HZ 50
#define INTERVAL_MS (1000 / FREQUENCY_HZ)
#define ACC_RANGE 1  // 0: -/+2G; 1: +/-4G

// convert factor g to m/s2 ==> [-32768, +32767] ==> [-2g, +2g]
#define CONVERT_G_TO_MS2 (9.81 / (16384.0 / (1. + ACC_RANGE)))

// convert factor to deg/s ==> [-32768, +32767] ==> [-250, +250] deg/s (default gyro range)
#define CONVERT_TO_DEG_S (1.0 / 131.0)

static unsigned long last_interval_ms = 0;

MPU6050 imu;
int16_t ax, ay, az;
int16_t gx, gy, gz;

void setup() {

  Serial.begin(115200);


  // initialize device
  Serial.println("Initializing I2C devices...");
  Wire.begin(4, 5);
  imu.initialize();
  delay(10);

  // verify connection
  if (imu.testConnection()) {
    Serial.println("IMU connected");
  }
  else {
    Serial.println("IMU Error");
  }
  delay(300);

  //Set MCU 6050 OffSet Calibration
  imu.setXAccelOffset(-4732);
  imu.setYAccelOffset(4703);
  imu.setZAccelOffset(8867);
  imu.setXGyroOffset(61);
  imu.setYGyroOffset(-73);
  imu.setZGyroOffset(35);

  /* Set full-scale accelerometer range.
     * 0 = +/- 2g
     * 1 = +/- 4g
     * 2 = +/- 8g
     * 3 = +/- 16g
     */
  imu.setFullScaleAccelRange(ACC_RANGE);
}

void loop() {

  if (millis() - last_interval_ms >= INTERVAL_MS) {
    // fixed-step schedule: advance by the period, not to "now", so the
    // average rate stays locked at FREQUENCY_HZ with no cumulative drift
    last_interval_ms += INTERVAL_MS;
    // resync after a stall to avoid a burst of catch-up samples
    if (millis() - last_interval_ms >= INTERVAL_MS) last_interval_ms = millis();

    // read raw accel/gyro measurements from device
    imu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // converting to m/s2
    float ax_m_s2 = ax * CONVERT_G_TO_MS2;
    float ay_m_s2 = ay * CONVERT_G_TO_MS2;
    float az_m_s2 = az * CONVERT_G_TO_MS2;

    // converting to deg/s
    float gx_deg_s = gx * CONVERT_TO_DEG_S;
    float gy_deg_s = gy * CONVERT_TO_DEG_S;
    float gz_deg_s = gz * CONVERT_TO_DEG_S;

    Serial.print(ax_m_s2);
    Serial.print("\t");
    Serial.print(ay_m_s2);
    Serial.print("\t");
    Serial.print(az_m_s2);
    Serial.print("\t");
    Serial.print(gx_deg_s);
    Serial.print("\t");
    Serial.print(gy_deg_s);
    Serial.print("\t");
    Serial.println(gz_deg_s);
  }
}