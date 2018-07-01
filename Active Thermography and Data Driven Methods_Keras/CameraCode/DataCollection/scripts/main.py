import socket, time, struct, array, os
import numpy as np
from matplotlib import pyplot as plt
from util import *




if __name__ == '__main__':
    # # Establishing Serial Connection with Teensy
    # while temp_dev == []:
    #     print "Setting up serial...",
    #     temp_dev = setup_serial(temp_dev_nm, baudrate)
    #     time.sleep(.05)
    # print "done"

    # Connecting to the port, listen to signal from thermal cam
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', PORT_NUM))

    # Check status of thermal cam
    client.recv(2)
    print "Ready Signal Recvd, Start Data Collection"

    # Check the name of the test material
    material = raw_input("Enter Material: ")


    for trial in range(NUM_TRIALS):
        print "Starting Trial: ", trial

        # Send starting signal to thermal cam
        path = os.path.join(DATA_PATH, material, 'trial%d/'%trial)
        if not os.path.exists(path):
            os.makedirs(path)
        client.send(path)


        # # Collecting Teensy reading
        # start_time = time.time()
        # prev_time = start_time
        # pas_temp = {}
        # pas_temp['time'] = []
        # pas_temp['data'] = []

        # dev_temp = setup_serial(temp_dev_nm, baudrate)
        # start_heating(dev_temp)

        # while time.time() - start_time < TRIAL_INTERVAL:

        #     # stop heating when heating time is reached
        #     if time.time() - start_time > HEAT_INTERVAL:
        #         stop_heating(dev_temp)

        #     try:
        #         rate = float(len(pas_temp['time']))/(Time_data[-1] - Time_data[0])
        #         check_time = np.clip(check_time + k_check_time*(1/rate - 1/freq), .0004, .0009)
        #     except:
        #         rate = 0

        #     if rate > freq:
        #         while (time.time() - prev_time) < (1/freq - check_time):
        #             print 'waiting'

        #     raw_temp_data = get_adc_data(temp_dev, temp_inputs) # list

        #     if raw_temp_data== [-1]: # Hack! [-1] is code for 'reset me'
        #         check = setup_serial(temp_dev_nm, baudrate)
        #         if check != []:
        #             dev_temp = check
        #             last_voltage_message = ' '
        #             last_supply_ki_message = " "
        #             print "reset temp serial"
        #     elif len(raw_temp_data) == temp_inputs:
        #         T = temperature([raw_temp_data[0]],3.3,8110.)[0]
        #         cur_time = time.time()
        #         pas_temp['time'] += [time.time() - start_time]
        #         pas_temp['data'] += [T]

        #         # smoothing
        #         if len(pas_temp['time']) > 100:
        #             pas_temp['data'][-1] = lfilter(fB,fA,pas_temp['data'])[-1]

        #         if cur_time - prev_time >= 1:
        #             prev_time = cur_time
        #             print '%.0f %.2f' % (pas_temp['time'][-1], pas_temp['data'][-1])

        # # Save teensy data
        # save_pickle(pas_temp, os.path.join(path, 'teensy_data.pkl'))

        # Wait for the thermal cam to completed data collection
        client.recv(2)

        print "Waiting for next trial"
        time.sleep(REST_INTERVAL)

    # Signal the thermal cam server to shut down
    print "Complete, Terminating"
    client.send('#')


