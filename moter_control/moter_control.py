#NOTE: This has to be run after sudo su to get access to the gpio pins and such, we would love to not have to do that, but have not found a way yet
# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time
import board
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685

i2c = board.I2C()  # uses board.SCL and board.SDA
# i2c = busio.I2C(board.GP1, board.GP0)    # Pi Pico RP2040

# Create a simple PCA9685 class instance.
pca.frequency = 60
SERVO_MIN_PWM = 1100
SERVO_MAX_PWM = 1900
class Servos:
    def __init__(self):
        pca = PCA9685(i2c)
        self.servos = []
        self.servos.append(servo.ContinuousServo(pca.channels[0], min_pulse=SERVO_MIN_PWM, max_pulse=SERVO_MAX_PWM))
        self.servos.append(servo.ContinuousServo(pca.channels[3], min_pulse=SERVO_MIN_PWM, max_pulse=SERVO_MAX_PWM))
        self.servos.append(servo.ContinuousServo(pca.channels[4], min_pulse=SERVO_MIN_PWM, max_pulse=SERVO_MAX_PWM))
        self.servos.append(servo.ContinuousServo(pca.channels[7], min_pulse=SERVO_MIN_PWM, max_pulse=SERVO_MAX_PWM))
        self.servos.append(servo.ContinuousServo(pca.channels[8], min_pulse=SERVO_MIN_PWM, max_pulse=SERVO_MAX_PWM))
        self.servos.append(servo.ContinuousServo(pca.channels[10], min_pulse=SERVO_MIN_PWM, max_pulse=SERVO_MAX_PWM))
        self.servos.append(servo.ContinuousServo(pca.channels[12], min_pulse=SERVO_MIN_PWM, max_pulse=SERVO_MAX_PWM))
        self.servos.append(servo.ContinuousServo(pca.channels[14], min_pulse=SERVO_MIN_PWM, max_pulse=SERVO_MAX_PWM))



    def set_servo(self, servo_num, speed):
        self.servos[servo_num].throttle = speed
    
    def stop_servo(self, servo_num):
        self.servos[servo_num].throttle = 0
    
    def stop_all_servos(self):
        for servo in self.servos:
            servo.throttle = 0
    
    def set_all_servos(self, speeds):
        for i in range(len(speeds)):
            self.servos[i].throttle = speeds[i]
    
    def test_run(self):
        for i in range(8):
            self.servos[i].throttle = .3
            time.sleep(1)
            self.servos[i].throttle = -.3
            time.sleep(1)
            self.servos[i].throttle = 0
            time.sleep(1)
# We sleep in import Servo

