#ifndef SOCKET_H
#define SOCKET_H

#include <string>
#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

class Socket
{
public:
    Socket();
    ~Socket();

    bool Create();
    bool Close();
    bool Bind(int port);
    bool Listen();
    bool Accept(Socket &newSocket);
    bool Connect(const std::string &address, int port);
    bool Send(const void *data, int length);
    bool Recv(void *data, int length);
    int Send(const void *data, int length, int flags);
    int Recv(void *data, int length, int flags);

private:
    const int INVALID = -1;
    int m_socket = INVALID;
};

#endif // SOCKET_H
