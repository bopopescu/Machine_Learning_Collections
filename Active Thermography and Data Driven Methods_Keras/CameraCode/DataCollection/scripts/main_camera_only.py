import socket, time, struct, array, os
import numpy as np
from matplotlib import pyplot as plt
from util import *




if __name__ == '__main__':
    # Connecting to the port, listen to signal from thermal cam
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', PORT_NUM))

    # Check status of thermal cam
    client.recv(2)
    print "Ready Signal Recvd, Start Data Collection"

    # Check the test material
    material = raw_input("Enter Material: ")


    for trial in range(NUM_TRIALS):
        print "Starting Trial: ", trial

        path = os.path.join(DATA_PATH, material, 'trial%d/'%trial)
        if not os.path.exists(path):
            os.makedirs(path)
        client.send(path)

        start = time.time()
        while time.time() - start < 30:
            print time.time() - start
            time.sleep(1)


        # Wait for the thermal cam to completed data collection
        client.recv(2)

        print "Waiting for next trial"
        time.sleep(REST_INTERVAL)

    # Signal the thermal cam server to shut down
    print "Complete, Terminating"
    client.send('#')


