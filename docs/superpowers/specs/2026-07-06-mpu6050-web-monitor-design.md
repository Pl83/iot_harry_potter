# MPU6050 Web Serial Monitor — Design

**Date:** 2026-07-06
**Status:** Approved (compressed flow — single-file scope, direct implementation approved by user)

## Goal

Live web visualization of the 6-axis MPU6050 data streamed by the ESP32-S3 Zero
(`sketch_jul6a.ino`) over native USB serial, with explicitly named curves.

## Architecture

- Single self-contained file: `web/mpu6050-monitor.html`. No external dependencies, no CDN.
- Opened directly in Chrome/Edge (Web Serial API required; Firefox unsupported).
- Firmware unchanged: it already emits `ax\tay\taz\tgx\tgy\tgz\n` at 115200 baud, 50 Hz.
  Acceleration in m/s², angular rate in °/s.

## Behavior

- **Connect** button → `navigator.serial.requestPort()`, open at 115200.
- Line-based parsing; a line must yield exactly 6 finite floats, otherwise it is ignored
  (boot messages like `IMU connected` are silently dropped).
- Two rolling charts (hand-drawn canvas), ~10 s window (500 samples at 50 Hz):
  1. **Accélération (m/s²)** — Accél X / Accél Y / Accél Z
  2. **Vitesse angulaire (°/s)** — Gyro X / Gyro Y / Gyro Z
- Legend per curve: explicit name, dedicated color, live numeric value.
- **Disconnect** button; USB unplug handled with a clear status message (no frozen page).

## Error handling

- Port already held by another process (Arduino IDE serial monitor) → readable error message.
- Browser without Web Serial → message telling the user to use Chrome/Edge.

## Testing

Manual: close Arduino IDE serial monitor, open page in Chrome, connect COM5,
verify ~9.81 on one accel axis at rest and near-zero gyro; move the board and watch curves.
