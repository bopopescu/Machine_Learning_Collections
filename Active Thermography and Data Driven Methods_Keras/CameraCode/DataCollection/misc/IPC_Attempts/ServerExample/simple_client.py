import socket, time

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 9999))
# print client.send('Hello world!'), 'bytes sent.'

while True:
    time.sleep(0.2)
    print 'Received message:', client.recv(1024)
    print client.send('Hello world!'), 'bytes sent.'
    print client.send('*'), 'bytes sent.'