#!/usr/bin/env python

import os
import sys
import serial
import math, numpy as np
import roslib; roslib.load_manifest('hrl_fabric_based_tactile_sensor')
import hrl_lib.util as ut
#import hrl_fabric_based_tactile_sensor.adc_publisher_node as apn
import rospy
import matplotlib.pyplot as plt

plt.ion()

import time
from scipy.signal import savgol_filter, lfilter, butter
from scipy.interpolate import interp1d
#from  hrl_fabric_based_tactile_sensor.map_thermistor_to_temperature import temperature

fB,fA = butter(2, 0.1, analog=False)

def setup_serial(dev_name, baudrate):
    try:
        serial_dev = serial.Serial(dev_name)
        if(serial_dev is None):
            raise RuntimeError("[%s]: Serial port %s not found!\n" % (rospy.get_name(), dev_name))

        serial_dev.setBaudrate(baudrate)
        serial_dev.setParity('N')
        serial_dev.setStopbits(1)
        serial_dev.write_timeout = .1
        serial_dev.timeout= 1

        serial_dev.flushOutput()
        serial_dev.flushInput()
        return serial_dev

    except serial.serialutil.SerialException as e:
        rospy.logwarn("[%s] Error initializing serial port %s", rospy.get_name(), dev_name)
        return []

def send_string(serial_dev, message):
    try:
        
        serial_dev.write(message) 
        serial_dev.flushOutput()
    except serial.serialutil.SerialException as e:
        print "Error sending string"
        
def get_adc_data(serial_dev, num_adc_inputs):     
    try:
        ln = serial_dev.readline()
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
    except:
        l = [-1] 
    return l    

def temperature(raw_data,Vsupp,Rref):
    raw_data = np.array(raw_data)
    Vref = 3.3
    Vin = raw_data/4095.0*Vref

    T1 = 288.15
    B = 3406
    R1 = 14827
    Vin[Vin <= 0] = .001
    RT = Rref*((Vsupp/Vin) - 1)
    RT[RT <= 0] = .001
    TC = (T1*B/np.log(R1/RT))/(B/np.log(R1/RT) - T1) - 273.15
    return TC.tolist()

def temperature_HT10K_Steinhart(data):

    R_ref = 9920
    V_plus = 3.3
    V_ref = 3.3
    R_T0 = R_ref*1.681/(V_plus-1.681)
    T0 = 273.15 + 23.70
    B = 3750.

    C1 = 1/3750. # 9.5594*10**(-4)
    C0 = 8.11e-4 # 1/(273.15+23.70) - C1*np.log(R_ref*1.681/(V_plus-1.681)) # 2.6181*10**(-4)
    temp_list = []
    for A_in in data:
        R_therm = R_ref*(((V_plus*4095)/(A_in*V_ref))-1)
        temp_list.append(1/(C1*np.log(R_therm) + C0)-273.15)
    return temp_list

def temperature_HT10K_exponential(data):
    return data
    R_ref = 9920
    V_plus = 3.3
    V_ref = 3.3
    R_T0 = R_ref*1.681/(V_plus-1.681)
    T0 = 273.15 + 23.70
    B = 3750.

    temp_list = []
    for A_in in data:
        R_therm = R_ref*(((V_plus*4095)/(A_in*V_ref))-1)
        temp_list.append( B / (np.log(R_therm/R_T0) + B/T0) - 273.15 )
    return temp_list


def run_exp(Material='test', sampleThickness=50, P_Gain=1e-3, I_Gain=0, D_Gain=8e-3, set_temp=35., cooling=False, duration=20., min_V=0.5, max_V=3.3, g1_V_setpoint=2.3):
    g1_temperature_setpoint_init = set_temp # 35 #C
    g2_temperature_setpoint = 0

    # P_Gain = .00000
    # I_Gain = 0.0006 # (was 0.0004) V per error in Celsuis, added at 100 hz 
    # D_Gain = 0.0000

    g1_error = 0
    g2_error = 0
    g1_sum_error = 0
    g2_sum_error = 0

    max_voltage = max_V
    min_voltage = min_V

    g1_voltage_setpoint = g1_V_setpoint #12 8.3 #Volts
    g2_voltage_setpoint = 0. # 3.3


    last_voltage_message = " "

    supply_ki_setpoint = .0001 # Integral gain for I controller on voltage supplies. Units are 12 bit unless duty PWM cycle values per mV of error. Added at 10khz. Good values of 0.0001 < supply_ki_set < .01 
    last_supply_ki_message = " "

    max_temp = 100 # C
    min_temp = 0
    temp_dev = []
    force_dev = []
    temp_dev_nm = '/dev/cu.teensy.s2672940' # thermal teensy serial number
    force_dev_nm = '/dev/cu.teensy.s2130840' # force teensy serial number
    baudrate = 115200

    ignore_trials = 3

    while temp_dev == [] or force_dev == []:
        print "Setting up serial...",
        temp_dev = setup_serial(temp_dev_nm, baudrate)
        force_dev = setup_serial(force_dev_nm, baudrate)
        time.sleep(.05)
    print "done"
    print ' '

    trial_index = 1
    trial_time = 5
 # seconds          
    up_time = 20                 

    wait_time = 1 # seconds 

    Fmax = 15  # Newto
    Fmin = 1 # Newtons
    Force_biased = False
    force_calib = .007 # Netwons per raw (0-4095) input

    dn = 1900 - sampleThickness*10 + 180 # us
    up = dn - 600 # - 300 #1000 # # us7.0,

    trials = 12
    tolerance = 0.2 #C
    #X = np.arange(11,16.1,tolerance*2).tolist()
    #desired_temp = X*trials #C
    desired_temp = [23]*trials

    g1_temperature_setpoints = set_temp + 0 * np.random.random(trials*2) # 30 + 5 * np.random.random(trials*2)

    save_data = False

    temp_inputs = 6
    force_inputs = 2
    freq = 200.
    check_time = .00067
    k_check_time = .002
    max_list_ln = int(freq*trial_time*4)
    while len(desired_temp) > 0 and not(rospy.is_shutdown()) and trial_index <= 1200:
        g1_temperature_setpoint = g1_temperature_setpoints[trial_index]

        voltage_message = "V 1:" + str(10*int(g1_voltage_setpoint*100)) + " 2:" + str(10*int(g2_voltage_setpoint*100)) + ' '
        send_string(temp_dev, voltage_message)
        print "moving actuator up"
        max_voltage = max_V
        min_voltage = min_V

        g1_voltage_setpoint = g1_V_setpoint
        z_set = up
        send_string(force_dev, str(z_set) + ' ')
        time.sleep(.1)
        send_string(force_dev, str(z_set) + ' ')
        time.sleep(.1)
        send_string(force_dev, str(z_set) + ' ')
        print "waiting for", wait_time, "seconds..."
        time.sleep(wait_time)
        print 't', 'F','  P ','  A ', '  Obj ', ' Vset', ' V'
        

        i = 0
        i_contact = 0
        t0 = time.time()
        t_last = t0
        Time_data = []
        F_data = []
        T0_data = []
        T1_data = []
        T2_data = []
        T3_data = []
        Noise = []
        STD = []
        restart = False
        rate = 0
        trial_start_time = 0
        G1_error_interval = 1
        G1_error = [10]*int(freq*G1_error_interval)
        G2_error = [10]*int(freq*G1_error_interval)
        #print "starting trial", trial_index
        max_period = 0
        while not(rospy.is_shutdown()) and not(restart):
            
            t_last = time.time()
            t = time.time() - t0
            Time_data.append(t)
            N = float(len(Time_data))
            try:
        
                period = (Time_data[-1] - Time_data[-2])
                max_period = max(period, max_period)
                rate = N/(Time_data[-1] - Time_data[0])
                check_time = np.clip(check_time + k_check_time*(1/rate - 1/freq), .0004, .0009)
            except:
                rate = 0    
            tic = time.time()
            # Send Data to Teensys; Control loop rate
            voltage_message = "V 1:" + str(10*int(g1_voltage_setpoint*100)) + " 2:" + str(10*int(g2_voltage_setpoint*100)) + ' '
            send_string(force_dev, str(z_set) + ' ')
            send_string(temp_dev, voltage_message)
            if rate > freq:        
                while (time.time() - t_last) < (1/freq - check_time):
                    'waiting'
            ## Get data from Temperature Teensy
            raw_temp_data = get_adc_data(temp_dev, temp_inputs) # list 
            #print raw_temp_data
            if raw_temp_data== [-1]: # Hack! [-1] is code for 'reset me'
                check = setup_serial(temp_dev_nm, baudrate)
                if check != []:
                    dev_temp = check
                    last_voltage_message = ' '
                    last_supply_ki_message = " "
                    print "reset temp serial"
            elif len(raw_temp_data) == temp_inputs:    
                T0 = temperature([raw_temp_data[0]],3.3,10000.)[0]
                T3 = temperature([raw_temp_data[1]],g1_voltage_setpoint,1000.)[0]
                T2 = temperature([raw_temp_data[2]],g2_voltage_setpoint,1000.)[0]
                T1 = temperature_HT10K_Steinhart([raw_temp_data[3]])[0] # temperature([raw_temp_data[3]],3.3,10000.)[0] 
                g1_voltage = raw_temp_data[4]/1000.0
                T0_data.append(T0) # append to list
                T1_data.append(T1) # append to list
                T2_data.append(T2) # append to list
                T3_data.append(T3) # append to list
                

                ################### For PID Optimization ###################

                # if len(T1_data) > freq*duration:
                #     send_string(force_dev, str(1000) + ' ')   
                #     time.sleep(1)
                #     return (T0_data, T1_data)
                
                # if cooling:
                #    if len(T1_data) > 100 and np.mean(T1_data[-100:]) < 32:
                #        print "T1: %f" % T1
                #        return
    
                #############################################################

                if i > 100:
                    T0_data[-1] = lfilter(fB,fA,T0_data)[-1]
                    T1_data[-1] = lfilter(fB,fA,T1_data)[-1]
                    T2_data[-1] = lfilter(fB,fA,T2_data)[-1]
                    T3_data[-1] = lfilter(fB,fA,T3_data)[-1]

                last_g1_error = np.mean(G1_error[-20:-10])
                g1_error = g1_temperature_setpoint - np.mean(T1_data[-10:])
                if t < 1.:
                    g1_error = g1_temperature_setpoint - T1_data[-1]
                ddt_g1_error = (np.mean(G1_error[-10:]) - last_g1_error)*(freq/10)
                
                G1_error.append(g1_error)
                G1_error.pop(0)
                g2_error = g2_temperature_setpoint - np.mean(T2_data[-1:])
                g1_sum_error = g1_sum_error + g1_error 
                g2_sum_error = g2_sum_error + g2_error
                
                voltage_message = "V 1:" + str(10*int(g1_voltage_setpoint*100)) + " 2:" + str(10*int(g2_voltage_setpoint*100)) + ' '
                supply_ki_message = 'K ' + str(supply_ki_setpoint) + ' ' 
                if not(supply_ki_message == last_supply_ki_message):
                    send_string(temp_dev, supply_ki_message)
                    last_supply_ki_message = supply_ki_message
            else:
                print "unable to get temp data"

            # Get data from Force Teensy
            raw_force_data = get_adc_data(force_dev,force_inputs) # list 
            if raw_force_data == [-1]: # Hack! [-1] is code for 'reset me'
                check = setup_serial(force_dev_nm, baudrate)
                if check != []:
                    dev_force = check
                    print "reset force serial"
            elif len(raw_force_data) == force_inputs:    
                f_raw = raw_force_data[0]
                z_raw = raw_force_data[1]
                z = -z_raw/4095.0*1000.0 + 2000.0
                if not(Force_biased):
                    if i == 0:
                        #print "Biasing Force Sensor"
                        f_bias = f_raw
                    elif i < 10:  
                        
                        f_bias = (f_bias + f_raw)/2
                    else:
                        Force_biased = True
                F = (f_raw - f_bias)*force_calib
                if F < Fmin:
                    F = 0
                F_data.append(F) 
                
                if F_data[-1] == 0:
                    g1_voltage_result = g1_voltage_setpoint + P_Gain*g1_error
                    if t > 1.:
                        g1_voltage_result = g1_voltage_setpoint + P_Gain*np.mean(G1_error[-10:]) + I_Gain*np.mean(G1_error) + D_Gain*ddt_g1_error

                    g1_voltage_setpoint = np.clip(g1_voltage_result, min_voltage,max_voltage)
                    g2_voltage_setpoint = 0 # np.clip(g2_voltage_setpoint + P_Gain*p_g2_error I_Gain*g2_error                , min_voltage,max_voltage)  
            else:
                print "unable to get force data"
            
            
            if len(Time_data) > max_list_ln:
                Time_data.pop(0)
                T0_data.pop(0)
                T1_data.pop(0)
                T2_data.pop(0)
                T3_data.pop(0)
                F_data.pop(0)
            # code to collect data
            #if np.mod(i,int(freq*.05)) == 0 and i > 0:
            #print ["%0.3f" % x for x in [T1_data[-1], g1_error, 1000*I_Gain*g1_error, 1000*D_Gain*ddt_g1_error, g1_voltage_setpoint]]
            
            if np.mod(i,int(freq)) == 0 and i > 0:
                print int(t), np.round(F_data[-1],1), np.round(np.mean(T0_data[-10:]),2), np.round(np.mean(T1_data[-10:]),2), np.round(np.mean(T3_data[-10:]),2), np.round(g1_voltage_setpoint,2), np.round(g1_voltage,2) #, np.round(z_set,0), np.round(z,0)
           
            if z_set == up and i > up_time*freq:
                a = 0
                while(a < len(desired_temp)):
                    if np.all(abs(np.array(G1_error)) < tolerance): # abs(np.mean(T3_data[-10:]) - desired_temp[a]) < tolerance and 
                        room_temp = np.mean(np.round(T0_data[:10],1))
                        desired_temp.pop(a)
                        z_set = dn
                        print "moving actuator down"
                        
                        t_down = time.time()
                        break
                    else:
                        a = a + 1
            
            if F_data[-1] == 0 and z_set == dn and time.time() - t_down > 20:
                print "Failed to detect object, Trying again"
                restart = True 
                save_data = False          
            if F_data[-1] > 5 and z_set == dn: # F_data[-1] > 0 and
                if i_contact == 0: # F_data[-2] == 0 and 
                    print "contact made"  

                    min_voltage = 0
                    max_voltage = 0
                    g1_voltage_setpoint = 0

                    trial_start_time = t
                    i_contact = i

                if (t - trial_start_time) > trial_time and not(trial_start_time == 0):
                    ##print len(T0_data), len(T1_data), len(T2_data), len(F_data)
                    restart = True
                    save_data = True
                    

            i=i+1


            #print time.time() - tic

        # WHEN TRIAL IS COMPLETE. REACH THIS POINT BY CALING restart = True 
            # at end of trial: plot and save data
        #time.sleep(3) 
           
        if save_data:
            if True:
                order = 3
                n_sample = 99
                T0_data = savgol_filter(T0_data,n_sample,order)
                T1_data = savgol_filter(T1_data, n_sample, order)
                T2_data = savgol_filter(T2_data, n_sample, order)
                T3_data = savgol_filter(T3_data, n_sample, order)
                Time_data = (np.array(Time_data) - trial_start_time).tolist()

                T0_func = interp1d(Time_data, T0_data)
                T1_func = interp1d(Time_data, T1_data)
                T2_func = interp1d(Time_data, T2_data)
                F_func = interp1d(Time_data, F_data)
                time_before_contact = 1.

                t = np.arange(-time_before_contact,trial_time + 1/freq,1/freq)
                T0_data = T0_func(t)
                T1_data = T1_func(t)
                T2_data = T2_func(t)
                F_data = F_func(t)


                directory = os.path.dirname(os.path.realpath(__file__)) + '/' + Material

                while os.path.exists(directory + '/trial_' + np.str(trial_index)):
                    trial_index = trial_index + 1
                
                directory = directory + '/trial_' + np.str(trial_index)
                
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    #print "created", directory

                
                A = np.array([t,T0_data, T1_data, T2_data, F_data]).T
                
                #i_save = np.argmin(abs(np.array(Time_data) + time_before_contact))
                #A = A[i_save:,:]
                ut.save_pickle(A, directory + "/room_temp_" + np.str(np.round(room_temp,2))+'.pkl')
                print directory + "/room_temp_" + str(np.round(room_temp,2))+'.pkl', "saved"
                print ' '
                
                if Material == 'test' or trial_index < 40:#show graphs                 
                
                   ## plt.figure()
                   ## plt.suptitle('Material Temperature='+' '+str(np.round(T0_data[i_contact],2))+'deg C') 
                   ## plt.subplot(4,1,1)
                   ## plt.plot(Time_data,T0_data,label='T0')
                    

                   ## plt.legend()
                   ## plt.axis([0,15,0,30])

                   ## plt.ylabel('Temp.($^\circ$C)')

                   ## frame = plt.gca()
                   ## frame.axes.get_xaxis().set_visible(False)

                    if True:
                        plt.subplot(2,1,1)

                        plt.plot(t,T1_data,'c') #,label='Active Thermal'


                        #plt.axis([0,15,0,35])

                        plt.ylabel('Temp.($^\circ$C)')
                        plt.legend()
                        frame = plt.gca()
                        #frame.axes.get_xaxis().set_visible(False)

                        plt.subplot(2,1,2)

                        plt.plot(t,T0_data,'r') #,label='Passive Thermal'

                        #plt.axis([0,15,0,35])

                        plt.ylabel('Temp.($^\circ$C)')
                        plt.legend()
                        frame = plt.gca()
                        #frame.axes.get_xaxis().set_visible(False)
                    else:
                        ignore_trials -= 1


                    #plt.subplot(4,1,3)
                    #plt.plot(Time_data, F_data,'r')
                    #plt.axis([0,15,0,30])
                    #plt.ylabel('Force (N)')
                    #frame = plt.gca()
                    #frame.axes.get_xaxis().set_visible(False)

                    ##plt.subplot(4,1,4)
                   ## print len(Time_data),len(z_data)
                    # plt.plot(t_data,z_data,'g')

                    #plt.axis([0,15,0,20])

                    ##plt.ylabel('Position (mm)')
                    ##plt.xlabel('Time (s)')
                    
                    plt.show(block=False)

                    plt.pause(0.05)
                trial_index = trial_index + 1
            else:
               print 'something went wrong. trying again'

    send_string(force_dev, str(1000) + ' ')   
    time.sleep(1)
    print "finished data collection"
       
def find_PID_params():
    err = float('inf')
    I = None
    for i in [5e-3]: # np.linspace(1e-4, 1e-3, 10):
        print "Testing P_Gain %f..." % i
        print "    Cooling..."
        run_exp(cooling=True, min_V=0., max_V=0., g1_V_setpoint=0.)
        print "    Experiment..."
        data = run_exp(P_Gain=8e-4, I_Gain=0, D_Gain=8e-3, set_temp=40, duration=60., Material='_')[2000:]
        _err = sum([abs(d-35) for d in data])
        if _err < err:
            err = _err
            I = i
            print "Updated err: %f, P_Gain: %f" % (err, I)

    print "    Cooling..."
    run_exp(cooling=True, min_voltage=0., max_voltage=0., g1_voltage_setpoint=0.)

    print "Final P_Gain: %f, with error: %f" % (I, err)

def save_trials_thermistor():
    counter = 0
    for i in range(30):
        data_heat = run_exp(P_Gain=8e-4, I_Gain=0, D_Gain=8e-3, set_temp=35, duration=60., Material='_')
        data_cool = run_exp(cooling=True, duration=60., Material='_')
        ut.save_pickle(data_heat, 'thermistor/%d.pkl' % counter)
        counter += 1
        ut.save_pickle(data_cool, 'thermistor/%d.pkl' % counter)
        counter += 1


if __name__ == '__main__':
    Material = sys.argv[1]
    sampleThickness = float(sys.argv[2])
    run_exp(Material=Material, sampleThickness=sampleThickness)
    # find_PID_params()
    # save_trials_thermistor()






            


