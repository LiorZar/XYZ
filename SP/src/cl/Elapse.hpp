#include <chrono>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <iomanip>

class Elapse
{
private:
    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
    struct Data
    {
        std::string name;
        std::vector<std::pair<std::string, int>> gpu;
    };
    std::map<TimePoint, Data> stamps;

public:
    Elapse(const std::string &name)
    {
        std::cout << "Start: " << name << std::endl;
        stamps[std::chrono::high_resolution_clock::now()] = {name, {}};
    }
    ~Elapse()
    {
        size_t total = 0;
        auto end = std::chrono::high_resolution_clock::now();
        auto it = stamps.begin();
        auto prev = it->first;
        ++it;
        for (; it != stamps.end(); ++it)
        {
            auto curr = it->first;
            const auto &data = it->second;
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(curr - prev).count();
            prev = curr;
            std::cout << data.name << ": " << TimeStr(duration) << std::endl;
            PrintData(data, total);
        }
        it = stamps.begin();
        auto start = it->first;
        const auto &data = it->second;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(prev - start).count();
        std::cout << data.name << ": " << TimeStr(duration) << std::endl;
        std::cout << "GPU Total: " << TimeStr(total) << std::endl;
    }

    void Stamp(const std::string &name)
    {
        stamps[std::chrono::high_resolution_clock::now()] = {name, {}};
    }
    void GStamp(const std::string &name, int duration)
    {
        auto &data = stamps.rbegin()->second;
        data.gpu.push_back({name, duration});
    }
    void PrintData(const Data &data, size_t &total) const
    {
        for (const auto &[name, duration] : data.gpu)
        {
            total += duration;
            std::cout << "\t" << name << ": " << TimeStr(duration) << std::endl;
        }
    }
    std::string TimeStr(size_t microseconds) const
    {
        std::stringstream ss;
        if (microseconds < 1000)
            ss << microseconds << "us";
        else if (microseconds < 1000000)
            ss << std::fixed << std::setprecision(2) << static_cast<double>(microseconds) / 1000.0 << "MS";
        else
            ss << std::fixed << std::setprecision(2) << static_cast<double>(microseconds) / 1000000.0 << "sec";
        return ss.str();
    }
};
