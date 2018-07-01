// #include <math.h>
int heartbeatPin = 13;

int Input1mv, Input2mv;
float Delta1mv, Delta2mv;

int outpin_source1 = 12;     /// change me 

void setup() {            
 Serial.begin(115200);  
 Serial.setTimeout(10);
 analogReference(DEFAULT);
 analogReadResolution(12);
 analogWriteResolution(12);

 pinMode(heartbeatPin,OUTPUT);
 pinMode(outpin_source1,OUTPUT);
 digitalWrite(heartbeatPin,HIGH);

 analogWriteFrequency(outpin_source1, 31250);
 digitalWrite(outpin_source1, HIGH);

}


int thermistor0_pin = 16;
// Future use for passive thermistor

float ki = .0001; // 0-4095 DC units per mv of error

void loop() {
        char command = (char)Serial.read();
        if (command=='G') {
          digitalWrite(outpin_source1, HIGH);
          digitalWrite(heartbeatPin, HIGH);
        } else if (command=='X') {
          digitalWrite(outpin_source1, LOW);
          digitalWrite(heartbeatPin, LOW);
        }
        //delayMicroseconds(500);

     
}







