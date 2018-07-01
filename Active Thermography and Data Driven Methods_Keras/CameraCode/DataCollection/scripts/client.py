import socket, time, struct, array, os
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
from util import *
plt.ion()
plt.figure()
fig = plt.imshow(np.zeros((lHeight, lWidth)))
plt.show()



def readall(filename):
    arr = array.array('H')
    print os.path.getsize(filename)/arr.itemsize
    arr.fromfile(open(filename, 'rb'), os.path.getsize(filename)/arr.itemsize)
    return np.array(arr, dtype='float64').reshape((lHeight, lWidth))

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 9999))


total_call = 0
total_time = 0.


while True:
    l = []
    start_time = time.time()





    # remain = 165888
    # allmsg = b""
    # while remain > 0:
    #     msg = client.recv(remain)
    #     remain -= len(msg)
    #     allmsg += msg
    # l = struct.unpack('%dH'%(82944), allmsg)



    client.recv(2)
    # l = readall("data.bin")





    time_elapsed = time.time() - start_time
    # # print 'Received message of length %d in %.4fs' % (len(l), time_elapsed)
    # # print l[:5], l[-5:], np.mean(l)

    # l -= np.min(l)
    # fig.set_data(l)
    # plt.draw()

    # Tell the server that all data is recieved
    path = 'test/test/'
    if not os.path.exists(path):
        os.makedirs(path)
    client.send(path)



    total_call += 1
    total_time += time_elapsed

    print "Average Time: ", total_time / total_call




