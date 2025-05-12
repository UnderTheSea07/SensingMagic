/*
 * Test sketch for data_logger1.py
 * Simulates ReSkin 4×MLX90393 sensor data
 */

unsigned long startTime;
bool streaming = false;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("#ReSkin Test Sensor");
  Serial.println("#Send 'g' to start streaming, 's' to stop");
  startTime = millis();
}

void loop() {
  // Check for commands
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'g') {
      streaming = true;
      Serial.println("#Streaming started");
    } else if (cmd == 's') {
      streaming = false;
      Serial.println("#Streaming stopped");
    }
  }
  
  // Generate simulated data if streaming
  if (streaming) {
    unsigned long currentTime = millis() - startTime;
    
    // Format MUST be exactly 17 values: timestamp T0 X0 Y0 Z0 T1 X1 Y1 Z1 T2 X2 Y2 Z2 T3 X3 Y3 Z3
    // Send values separated by spaces, not tabs
    Serial.print(currentTime);  // Timestamp in ms
    
    // Simulate 4 sensors with T, X, Y, Z values
    for (int sensor = 0; sensor < 4; sensor++) {
      float temp = 25.0 + random(-10, 10) / 10.0;  // Temperature around 25°C
      
      // Generate sine waves with different phases for each sensor
      float angle = (currentTime / 1000.0) + (sensor * PI / 2);
      // Scale down the values to match expected range
      float x = sin(angle) * 100;
      float y = cos(angle) * 100;
      float z = sin(angle * 2) * 50;
      
      Serial.print(" ");
      Serial.print(temp);
      Serial.print(" ");
      Serial.print(x);
      Serial.print(" ");
      Serial.print(y);
      Serial.print(" ");
      Serial.print(z);
    }
    
    Serial.println();
    delay(100);  // Send data 10 times per second
  }
} 