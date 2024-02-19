#ifndef BINBUFFER_H
#define BINBUFFER_H

#include "Utils.hpp"
#include <vector>
#include <string>
#include <cstring>

class BinBuffer
{
private:
    std::vector<char> buffer;
    size_t readPos;

public:
    BinBuffer() : readPos(0) {}

    size_t size() const { return buffer.size(); }
    const char *const_pointer() const { return buffer.data(); }
    void clear()
    {
        buffer.clear();
        readPos = 0;
    }

    void WriteInt(int value)
    {
        char *data = reinterpret_cast<char *>(&value);
        buffer.insert(buffer.end(), data, data + sizeof(int));
    }

    int ReadInt()
    {
        int value;
        std::memcpy(&value, &buffer[readPos], sizeof(int));
        readPos += sizeof(int);
        return value;
    }

    void WriteString(const std::string &value)
    {
        WriteInt(value.size());
        buffer.insert(buffer.end(), value.begin(), value.end());
    }
    std::string ReadString()
    {
        int size = ReadInt();
        std::string value(buffer.begin() + readPos, buffer.begin() + readPos + size);
        readPos += size;
        return value;
    }

    void WriteFloat(float value)
    {
        char *data = reinterpret_cast<char *>(&value);
        buffer.insert(buffer.end(), data, data + sizeof(float));
    }

    float ReadFloat()
    {
        float value;
        std::memcpy(&value, &buffer[readPos], sizeof(float));
        readPos += sizeof(float);
        return value;
    }

    template <typename T>
    void WriteVector(const std::vector<T> &value)
    {
        WriteInt(value.size());
        for (const auto &element : value)
        {
            Write(element);
        }
    }

    template <typename T>
    std::vector<T> ReadVector()
    {
        int size = ReadInt();
        std::vector<T> value;
        value.reserve(size);
        for (int i = 0; i < size; ++i)
        {
            value.push_back(Read<T>());
        }
        return value;
    }

    template <typename T>
    void Write(const T &value)
    {
        char *data = reinterpret_cast<char *>(&value);
        buffer.insert(buffer.end(), data, data + sizeof(T));
    }

    template <typename T>
    T Read()
    {
        T value;
        std::memcpy(&value, &buffer[readPos], sizeof(T));
        readPos += sizeof(T);
        return value;
    }
    void WriteNode(const XNode &node)
    {
        WriteInt(node.size());
        for (const auto &pair : node)
        {
            WriteString(pair.first);
            WriteString(pair.second);
        }
    }
    bool ReadNode(XNode &node)
    {
        int size = ReadInt();
        for (int i = 0; i < size; ++i)
        {
            std::string key = ReadString();
            std::string value = ReadString();
            node[key] = value;
        }
        return true;
    }
};

#endif // BINBUFFER_H