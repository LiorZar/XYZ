#ifndef WS_PROCESSOR_H
#define WS_PROCESSOR_H

#include "Utils.hpp"
#include "BinBuffer.hpp"

class WSProcessor
{
public:
    static bool ProcessIncomingMessage(const BinBuffer &message);
    static bool ProcessOutgoingMessage(const BinBuffer &message);
    static bool ProcessHandshake(BinBuffer &request);

private:
    static bool ParseHandshake(const BinBuffer &message, XNode &node);
};

#endif // WS_PROCESSOR_H