#!/usr/bin/env python

import os
import sys
import serial
from glob import glob
import math, numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import savgol_filter, lfilter, butter
from scipy.interpolate import interp1d

fB,fA = butter(2, 0.1, analog=False)
temp_dev_nm = '/dev/cu.usbmodem2673411' # thermal teensy serial number
baudrate = 115200
temp_dev = []
temp_inputs = 2
freq = 100.
check_time = .00067
k_check_time = .002

def setup_serial(dev_name, baudrate):
    try:
        serial_dev = serial.Serial(dev_name)
        if(serial_dev is None):
            raise RuntimeError("Serial port %s not found!\n" % (dev_name))

        serial_dev.baudrate = baudrate
        serial_dev.parity = 'N'
        serial_dev.stopbits = 1
        serial_dev.write_timeout = .1
        serial_dev.timeout= 1.

        serial_dev.flushOutput()
        serial_dev.flushInput()
        return serial_dev

    except serial.serialutil.SerialException as e:
        print "Serial port %s not found!\n" % (dev_name)
        return []

def send_string(serial_dev, message):
    try:
        serial_dev.write(message)
        serial_dev.flushOutput()
    except serial.serialutil.SerialException as e:
        print "Error sending string"

def get_adc_data(serial_dev, num_adc_inputs):

    ln = serial_dev.readline()
    print serial_dev, ln
    if not ln:
        print 'not received'
        return []

    #serial_dev.flushInput()
    try:
        l = map(int, ln.split(','))
    except ValueError:
        serial_dev.flush()
        l = []
    if len(l) != num_adc_inputs:
        serial_dev.flush()
        l = []
    else:
        return l

    # print 'passed'
    return l

def temperature(raw_data,Vsupp,Rref):
    raw_data = np.array(raw_data)
    # Vref = 3.3
    # Vin = raw_data/4095.0*Vref

    T1 = 288.15
    B = 3406
    R1 = 14827
    Vin[Vin <= 0] = .001
    RT = Rref*(Vsupp/raw_data - 1)
    RT[RT <= 0] = .001
    TC = (T1*B/np.log(R1/RT))/(B/np.log(R1/RT) - T1) - 273.15
    return TC.tolist()

def start_heating(temp_dev):
    send_string(temp_dev, 'G')

def stop_heating(temp_dev):
    send_string(temp_dev, 'X')


if __name__ == '__main__':
    while temp_dev == []:
        print "Setting up serial...",
        temp_dev = setup_serial(temp_dev_nm, baudrate)
        print temp_dev

        time.sleep(.05)
    print "done"
    print ' '

    # start_heating(temp_dev)
    # time.sleep(5)
    # stop_heating(temp_dev)
    # time.sleep(1)

    start_time = time.time()
    prev_time = start_time
    pas_temp = {}
    pas_temp['time'] = []
    pas_temp['data'] = []

    # while True:
    #     print len(temp_dev.readline())

    while True: # while not(rospy.is_shutdown()):

        try:
            print pas_temp['time']
            rate = float(len(pas_temp['time']))/(pas_temp['time'][-1] - pas_temp['time'][0])
            check_time = np.clip(check_time + k_check_time*(1/rate - 1/freq), .0004, .0009)
            # else:
            #     rate = 0
        except Exception as e:
            print e
            rate = 0

        if rate > freq:
            while (time.time() - prev_time) < (1/freq - check_time):
                print 'waiting'

        # print rate

        raw_temp_data = get_adc_data(temp_dev, temp_inputs) # list

        print raw_temp_data, temp_inputs
        if raw_temp_data== [-1]: # Hack! [-1] is code for 'reset me'
            check = setup_serial(temp_dev_nm, baudrate)
            if check != []:
                dev_temp = check
                last_voltage_message = ' '
                last_supply_ki_message = " "
                print "reset temp serial"
        elif len(raw_temp_data) == temp_inputs:
            T = temperature([raw_temp_data[0]],raw_temp_data[1],8110.)[0]
            cur_time = time.time()
            pas_temp['time'] += [time.time() - start_time]
            pas_temp['data'] += [T]

            # smoothing
            if len(pas_temp['time']) > 100:
                pas_temp['data'][-1] = lfilter(fB,fA,pas_temp['data'])[-1]

            if cur_time - prev_time >= 1:
                prev_time = cur_time
                print '%.0f %.2f' % (pas_temp['time'][-1], pas_temp['data'][-1])



