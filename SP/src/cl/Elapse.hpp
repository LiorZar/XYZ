#include <chrono>
#include <iostream>
#include <string>
#include <map>

class Elapse
{
private:
    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
    std::map<TimePoint, std::string> stamps;

public:
    Elapse(const std::string &name)
    {
        std::cout << "Start: " << name << std::endl;
        stamps[std::chrono::high_resolution_clock::now()] = name;
    }
    ~Elapse()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto it = stamps.begin();
        auto prev = it->first;
        ++it;
        for (; it != stamps.end(); ++it)
        {
            auto curr = it->first;
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(curr - prev).count();
            prev = curr;
            std::cout << it->second << ": " << duration << "us" << std::endl;
        }
        it = stamps.begin();
        auto start = it->first;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(prev - start).count();
        std::cout << it->second << ": " << duration << "us" << std::endl;
    }

    void Stamp(const std::string &name)
    {
        stamps[std::chrono::high_resolution_clock::now()] = name;
    }
};
