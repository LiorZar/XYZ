#include "Socket.h"
#include <iostream>

Socket::Socket()
{
    // Constructor implementation
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Socket::~Socket()
{
    Close();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Socket::Create()
{
    if (INVALID != m_socket)
        Close();

    m_socket = socket(AF_INET, SOCK_STREAM, 0);

    if (INVALID == m_socket)
        return false;

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Socket::Close()
{
    if (INVALID != m_socket)
    {
        closesocket(m_socket);
        m_socket = INVALID;
    }
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Socket::Bind(int port)
{
    if (INVALID == m_socket)
        return false;

    SOCKADDR_IN serverAddress;
    std::memset(&serverAddress, 0, sizeof(serverAddress));
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons((short)port);
    serverAddress.sin_addr.s_addr = htonl(INADDR_ANY);

    if (SOCKET_ERROR == bind(m_socket, (PSOCKADDR)&serverAddress, sizeof(serverAddress)))
        return false;

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Socket::Listen()
{
    if (INVALID == m_socket)
        return false;

    if (SOCKET_ERROR == listen(m_socket, SOMAXCONN))
        return false;

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Socket::Accept(Socket &clientSocket)
{
    if (INVALID == m_socket)
        return false;

    clientSocket.m_socket = accept(m_socket, nullptr, nullptr);
    if (clientSocket.m_socket == INVALID)
        return false;

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Socket::Connect(const std::string &ipAddress, int port)
{
    if (INVALID == m_socket)
        return false;

    SOCKADDR_IN address;
    std::memset(&address, 0, sizeof(address));

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr(ipAddress.c_str());
    address.sin_port = htons((short)port);

    if (SOCKET_ERROR != connect(m_socket, (PSOCKADDR)&address, sizeof(SOCKADDR)))
        return true;

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
int Socket::Send(const void *buffer, int length, int flags)
{
    if (INVALID == m_socket)
        return 0;

    return send(m_socket, (const char *)buffer, length, flags);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
int Socket::Recv(void *buffer, int length, int flags)
{
    if (INVALID == m_socket)
        return 0;

    return recv(m_socket, (char *)buffer, length, flags);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Socket::Send(const void *buffer, int length)
{
    if (INVALID == m_socket)
        return false;

    if (length <= 0)
        return true;

    int bytesSent = 0;
    const char *ptr = (const char *)buffer;
    do
    {
        bytesSent = send(m_socket, ptr, length, 0);
        ptr += bytesSent;
        length -= bytesSent;

    } while (bytesSent > 0 && length > 0);

    if (length <= 0)
        return true;

    Close();

    return false;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Socket::Recv(void *buffer, int length)
{
    if (INVALID == m_socket)
        return false;

    if (length <= 0)
        return true;

    int bytesRecv = 0;
    char *ptr = (char *)buffer;
    do
    {
        bytesRecv = recv(m_socket, ptr, length, 0);
        ptr += bytesRecv;
        length -= bytesRecv;

    } while (bytesRecv > 0 && length > 0);

    if (length <= 0)
        return true;

    Close();

    return false;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
