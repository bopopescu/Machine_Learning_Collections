#include <iostream>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <unistd.h>

#include <fstream>

using namespace std;

#define PORT_NUM 9999

struct connection {
    int server;
    int client;
} connection_t;

struct connection establish_connection();
void stop_connection(struct connection conn);


int main()
{
    struct connection conn;
    conn = establish_connection();
    int server, client;
    server = conn.server;
    client = conn.client;

    uint16_t foo [82944];

    for (int counter = 0; counter < 82944; counter++)
    {
        foo[counter] = counter % 65535;
    }

    while (server > 0)
    {
        cout << "Sending " << sizeof(foo) << " bytes" << endl;



        ofstream myFile ("data.bin", ios::out | ios::binary);
        myFile.write ( (char*)foo, 82944*sizeof(uint16_t) );
        myFile.flush();
        myFile.close();


        char* buffer[1024];
        send(server, buffer, 1024, 0);
        recv(server, buffer, 1024, 0);
    }

    stop_connection(conn);

    return 0;
}

struct connection establish_connection()
{
    struct connection conn;
    int server, client;
    int portNum = PORT_NUM;

    struct sockaddr_in server_addr;
    socklen_t size;

    client = socket(AF_INET, SOCK_STREAM, 0);

    if (client < 0)
    {
        cout << "\nError establishing socket..." << endl;
        exit(1);
    }

    cout << "\n=> Socket server has been created..." << endl;

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htons(INADDR_ANY);
    server_addr.sin_port = htons(portNum);


    if ((::bind(client, (struct sockaddr*)&server_addr,sizeof(server_addr))) < 0)
    {
        cout << "=> Error binding connection, the socket has already been established..." << endl;
        exit(1);
    }

    size = sizeof(server_addr);
    cout << "=> Looking for clients..." << endl;

    listen(client, 1);

    int clientCount = 1;
    server = accept(client,(struct sockaddr *)&server_addr,&size);

    // first check if it is valid or not
    if (server < 0)
        cout << "=> Error on accepting..." << endl;

    cout << "=> Connected with the client #" << clientCount << ", you are good to go..." << endl;

    conn.server = server;
    conn.client = client;
    return conn;
}

void stop_connection(struct connection conn)
{
    int server, client;
    server = conn.server;
    client = conn.client;

    close(server);
    cout << "\nGoodbye..." << endl;

    close(client);
}