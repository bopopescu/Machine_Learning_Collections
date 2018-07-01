// *****************************************************************************
//
//    Copyright (c) 2013, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

//
// Shows how to use a PvPipeline object to acquire images from a GigE Vision or
// USB3 Vision device.
//

#include <eBUS/PvSampleUtils.h>
#include <eBUS/PvDevice.h>
#include <eBUS/PvDeviceGEV.h>
#include <eBUS/PvDeviceU3V.h>
#include <eBUS/PvStream.h>
#include <eBUS/PvStreamGEV.h>
#include <eBUS/PvStreamU3V.h>
#include <eBUS/PvPipeline.h>
#include <eBUS/PvBuffer.h>
#include <eBUS/PvBufferWriter.h>

#include <iostream>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <unistd.h>

#include <fstream>

// using namespace std;

struct connection {
    int server;
    int client;
} connection_t;


PV_INIT_SIGNAL_HANDLER();

#define BUFFER_COUNT ( 16 )
#define IMAGE_SAVE_LOC ""

#define PORT_NUM 9999
#define BUF_SIZE 1024
#define TRIAL_TIME 10200
#define FPS 60


///
/// Function Prototypes
///
PvDevice *ConnectToDevice( const PvString &aConnectionID );
PvStream *OpenStream( const PvString &aConnectionID );
void ConfigureStream( PvDevice *aDevice, PvStream *aStream );
PvPipeline* CreatePipeline( PvDevice *aDevice, PvStream *aStream );
void AcquireImages( struct connection conn, PvDevice *aDevice, PvStream *aStream, PvPipeline *aPipeline );

struct connection establish_connection();
void stop_connection(struct connection conn);

//
// Main function
//
int main()
{
    struct connection conn;
    conn = establish_connection();
    int server, client;
    server = conn.server;
    client = conn.client;
    
    
    PvDevice *lDevice = NULL;
    PvStream *lStream = NULL;

    PV_SAMPLE_INIT();

    cout << "PvPipelineSample:" << endl << endl;

    PvString lConnectionID;
    if ( PvSelectDevice( &lConnectionID ) )
    {
        lDevice = ConnectToDevice( lConnectionID );
        if ( lDevice != NULL )
        {
            lStream = OpenStream( lConnectionID );
            if ( lStream != NULL )
            {
                PvPipeline *lPipeline = NULL;

                ConfigureStream( lDevice, lStream );
                lPipeline = CreatePipeline( lDevice, lStream );
                if( lPipeline )
                {
                    AcquireImages(conn, lDevice, lStream, lPipeline );
                    delete lPipeline;
                }

                // Close the stream
                cout << "Closing stream" << endl;
                lStream->Close();
                PvStream::Free( lStream );
            }

            // Disconnect the device
            cout << "Disconnecting device" << endl;
            lDevice->Disconnect();
            PvDevice::Free( lDevice );
        }
    }

    cout << endl;
    // cout << "<press a key to exit>" << endl;
    // PvWaitForKeyPress();
    
    stop_connection(conn);

    PV_SAMPLE_TERMINATE();

    return 0;
}

PvDevice *ConnectToDevice( const PvString &aConnectionID )
{
    PvDevice *lDevice;
    PvResult lResult;

    // Connect to the GigE Vision or USB3 Vision device
    cout << "Connecting to device." << endl;
    lDevice = PvDevice::CreateAndConnect( aConnectionID, &lResult );
    if ( lDevice == NULL )
    {
        cout << "Unable to connect to device." << endl;
    }

    return lDevice;
}

PvStream *OpenStream( const PvString &aConnectionID )
{
    PvStream *lStream;
    PvResult lResult;

    // Open stream to the GigE Vision or USB3 Vision device
    cout << "Opening stream from device." << endl;
    lStream = PvStream::CreateAndOpen( aConnectionID, &lResult );
    if ( lStream == NULL )
    {
        cout << "Unable to stream from device." << endl;
    }

    return lStream;
}

void ConfigureStream( PvDevice *aDevice, PvStream *aStream )
{
    // If this is a GigE Vision device, configure GigE Vision specific streaming parameters
    PvDeviceGEV* lDeviceGEV = dynamic_cast<PvDeviceGEV *>( aDevice );
    if ( lDeviceGEV != NULL )
    {
        PvStreamGEV *lStreamGEV = static_cast<PvStreamGEV *>( aStream );

        // Negotiate packet size
        lDeviceGEV->NegotiatePacketSize();

        // Configure device streaming destination
        lDeviceGEV->SetStreamDestination( lStreamGEV->GetLocalIPAddress(), lStreamGEV->GetLocalPort() );
    }
}

PvPipeline *CreatePipeline( PvDevice *aDevice, PvStream *aStream )
{
    // Create the PvPipeline object
    PvPipeline* lPipeline = new PvPipeline( aStream );

    if ( lPipeline != NULL )
    {
        // Reading payload size from device
        uint32_t lSize = aDevice->GetPayloadSize();

        // Set the Buffer count and the Buffer size
        lPipeline->SetBufferCount( BUFFER_COUNT );
        lPipeline->SetBufferSize( lSize );
    }

    return lPipeline;
}

void AcquireImages(struct connection conn, PvDevice *aDevice, PvStream *aStream, PvPipeline *aPipeline )
{
    int server, client;
    server = conn.server;
    client = conn.client;
    bool stopStream = false;
    
    // Get device parameters need to control streaming
    PvGenParameterArray *lDeviceParams = aDevice->GetParameters();

    // Map the GenICam AcquisitionStart and AcquisitionStop commands
    PvGenCommand *lStart = dynamic_cast<PvGenCommand *>( lDeviceParams->Get( "AcquisitionStart" ) );
    PvGenCommand *lStop = dynamic_cast<PvGenCommand *>( lDeviceParams->Get( "AcquisitionStop" ) );

    // Note: the pipeline must be initialized before we start acquisition
    cout << "Starting pipeline" << endl;
    aPipeline->Start();

    // Get stream parameters
    PvGenParameterArray *lStreamParams = aStream->GetParameters();

    // Map a few GenICam stream stats counters
    PvGenFloat *lFrameRate = dynamic_cast<PvGenFloat *>( lStreamParams->Get( "AcquisitionRate" ) );
    PvGenFloat *lBandwidth = dynamic_cast<PvGenFloat *>( lStreamParams->Get( "Bandwidth" ) );

    // Enable streaming and send the AcquisitionStart command
    cout << "Enabling streaming and sending AcquisitionStart command." << endl;
    aDevice->StreamEnable();
    lStart->Execute();

    char lDoodle[] = "|\\-|-/";
    int lDoodleIndex = 0;
    double lFrameRateVal = 0.0;
    double lBandwidthVal = 0.0;

    // Acquire images until the user instructs us to stop.
    cout << endl << "<press a key to stop streaming>" << endl;
    while ( !stopStream )
    {
        
        if (server > 0)
        {
            char sig[10] = "*";
            send(server, sig, 2, 0); // Just for signaling purpose
            
            
            char response_buffer[BUF_SIZE];
            memset(&response_buffer[0], 0, BUF_SIZE);
            recv(server, response_buffer, BUF_SIZE, 0);
            cout << response_buffer << " " << endl;
            if (*response_buffer == '#')
            {
                stopStream = true;
            }
            else
            {
                std::chrono::time_point<std::chrono::system_clock> start_t = std::chrono::system_clock::now();
                
                while( chrono::duration_cast<chrono::milliseconds>(std::chrono::system_clock::now() - start_t).count()  < TRIAL_TIME )
                {
                    cout << "Time Elapsed: " << chrono::duration_cast<chrono::milliseconds>(std::chrono::system_clock::now() - start_t).count() << endl;
                    
                    PvBuffer *lBuffer = NULL;
                    PvResult lOperationResult;
                    PvBufferWriter lBufferWriter;
                    
                    // Retrieve next buffer
                    PvResult lResult = aPipeline->RetrieveNextBuffer( &lBuffer, 1000, &lOperationResult );
                    if ( lResult.IsOK() )
                    {
                        // cout << "lResult Ok" << endl;
                        if ( lOperationResult.IsOK() )
                        {
                            PvPayloadType lType;
                            
                            //
                            // We now have a valid buffer. This is where you would typically process the buffer.
                            // -----------------------------------------------------------------------------------------
                            // ...
                            
                            lFrameRate->GetValue( lFrameRateVal );
                            lBandwidth->GetValue( lBandwidthVal );
                            
                            // If the buffer contains an image, display width and height.
                            uint32_t lWidth = 0, lHeight = 0;
                            lType = lBuffer->GetPayloadType();
                            
                            cout << fixed << setprecision( 1 );
                            cout << lDoodle[ lDoodleIndex ];
                            cout << " BlockID: " << uppercase << hex << setfill( '0' ) << setw( 16 ) << lBuffer->GetBlockID();
                            if ( lType == PvPayloadTypeImage )
                            {
                                // Get image specific buffer interface.
                                PvImage *lImage = lBuffer->GetImage();
                                
                                // Read width, height.
                                lWidth = lImage->GetWidth();
                                lHeight = lImage->GetHeight();
                                cout << "  W: " << dec << lWidth << " H: " << lHeight;
                                // cout << " pixel bits: " << lImage->GetBitsPerPixel() << " pixel type: " << lImage->GetPixelType();
                                
                                if (lBuffer->GetBlockID() % ((int)(60/FPS)) ==0) {
                                    
                                    // Save Image
                                    // char filename[]= IMAGE_SAVE_LOC;
                                    // std::string s=std::to_string(lBuffer->GetBlockID());
                                    char filename[BUF_SIZE];
                                    memset(&filename[0], 0, BUF_SIZE);
                                    strcpy(filename, response_buffer);
                                    std::string s=std::to_string( chrono::duration_cast<chrono::milliseconds>(std::chrono::system_clock::now() - start_t).count() );
                                    std::string path(filename);
                                    std::size_t found = path.find_last_of("/");
                                    std::string filetype(".bin");
                                    std::string complete = path.substr(0,found+1) + s + filetype;

                                    cout << "Path: " << complete << endl;
                                    lBufferWriter.Store(lBuffer,complete.c_str(),PvBufferFormatRaw);
                                    memset(filename, 0, BUF_SIZE);
                                    
                                }
                                
                            }
                            else {
                                cout << " (buffer does not contain image)";
                            }
                            cout << "  " << lFrameRateVal << " FPS  " << ( lBandwidthVal / 1000000.0 ) << " Mb/s   \r";
                        }
                        else
                        {
                            // Non OK operational result
                            cout << lDoodle[ lDoodleIndex ] << " " << lOperationResult.GetCodeString().GetAscii() << "\r";
                        }
                        
                        // Release the buffer back to the pipeline
                        aPipeline->ReleaseBuffer( lBuffer );
                    }
                    else
                    {
                        // Retrieve buffer failure
                        cout << lDoodle[ lDoodleIndex ] << " " << lResult.GetCodeString().GetAscii() << "\r";
                    }
                    
                    ++lDoodleIndex %= 6;
                }
                
                
            }
            
            memset(response_buffer, 0, BUF_SIZE);
            
            
        }
        
    }

    // PvGetChar(); // Flush key buffer for next stop.
    cout << endl << endl;

    // Tell the device to stop sending images.
    cout << "Sending AcquisitionStop command to the device" << endl;
    lStop->Execute();

    // Disable streaming on the device
    cout << "Disable streaming on the controller." << endl;
    aDevice->StreamDisable();

    // Stop the pipeline
    cout << "Stop pipeline" << endl;
    aPipeline->Stop();
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
