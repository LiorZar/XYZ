#include "WSProcessor.h"

//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool WSProcessor::ProcessIncomingMessage(const BinBuffer &message)
{
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool WSProcessor::ProcessOutgoingMessage(const BinBuffer &message)
{
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool WSProcessor::ProcessHandshake(BinBuffer &src)
{
    // XNode request;
    // request["msg"] = "websocket";
    // request["codes"] = "handshake";

    // if (false == ParseHandshake(src, request))
    //     return false;

    // src.clear();
    // src.WriteNode(request);

    // XStr key;
    // if (false == request("Sec-WebSocket-Key", key))
    //     return false;
    // key += "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

    // crypt::SHA1 sha;
    // XBin digest;
    // digest.resize(20);
    // ui32 *message_digest = (ui32 *)digest.pointer();

    // sha.Reset();
    // sha << key.c_str();

    // sha.Result(message_digest);

    // // convert sha1 hash bytes to network byte order because this sha1
    // //  library works on ints rather than bytes
    // for (int i = 0; i < 5; ++i)
    //     message_digest[i] = htonl(message_digest[i]);

    // key = XStr::EncodeBase64(digest);

    // str += "HTTP/1.1 101 Switching Protocols\r\n";
    // str += "Connection: Upgrade\r\n";
    // str += "Sec-WebSocket-Accept: " + key + "\r\n";
    // str += "Upgrade: websocket\r\n";
    // str += "\r\n";

    // return true;
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool WSProcessor::ParseHandshake(const BinBuffer &message, XNode &node)
{
    if (message.size() < 0)
        return false;

    std::string data = std::string(message.const_pointer());

    XParams lines = Utils::Split(data, "\r\n");
    if (lines.size() < 1)
        return false;

    const auto &status = lines[0];
    XParams args = Utils::Split(status, " ");
    if (args.size() < 3)
        return false;

    node["method"] = args[0];
    node["uri"] = args[1];
    node["version"] = args[2];

    for (auto i = 1U; i < lines.size(); ++i)
    {
        const auto &line = lines[i];

        args = Utils::Split(line, ": ");
        if (2 != args.size())
            continue;

        node[args[0]] = args[1];
    }

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
