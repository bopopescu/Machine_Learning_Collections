import socket, time, struct

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 9999))


total_call = 0
total_time = 0.


while True:
    l = []
    start_time = time.time()






    # counter = 0
    # while counter < 165888:
    #     msg = client.recv(2)
    #     res = struct.unpack('H', msg)
    #     l.append(res[0])
    #     counter+=2



    remain = 165888
    allmsg = b""
    while remain > 0:
        msg = client.recv(remain)
        remain -= len(msg)
        allmsg += msg
    # for i in range(165888/2):
    #     res = struct.unpack('H', allmsg[2*i:2*(i+1)])
    #     l.append(res[0])
    l = range(100)
    print len(msg)









    time_elapsed = time.time() - start_time
    print 'Received message of length %d in %.4fs' % (len(l), time_elapsed)
    print l[:5], l[-5:]

    # Tell the server that all data is recieved
    client.send('lol')



    total_call += 1
    total_time += time_elapsed
    print "Average Time: ", total_time / total_call




