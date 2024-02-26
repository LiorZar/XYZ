#include "defines.h"

NAMESPACE_BEGIN(cu);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
class Elapse
{
private:
    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
    struct Data
    {
        std::string name;
        float gpu = 0.f;
        float cpu = 0.f;
        float count = 0.f;
    };
    std::vector<TimePoint> cpu;
    std::vector<std::string> stamps;
    std::vector<cudaEvent_t> events;

public:
    Elapse(const std::string &name, int eventCount = 10)
    {
        Stamp(name);
    }
    ~Elapse()
    {
        Stamp("End");
        cudaEventSynchronize(events[stamps.size() - 1]);

        int index = 0;
        std::map<std::string, int> name2index;
        std::vector<Data> times;
        float milliseconds = 0, microseconds = 0;
        for (auto i = 1U; i < stamps.size(); ++i)
        {
            auto &start = events[i - 1];
            auto &end = events[i];
            cudaEventElapsedTime(&milliseconds, start, end);
            microseconds = (float)std::chrono::duration_cast<std::chrono::microseconds>(cpu[i] - cpu[i - 1]).count();
            index = name2index[stamps[i]];
            if (index == 0)
            {
                index = (int)times.size();
                name2index[stamps[i]] = index;
                times.push_back({stamps[i]});
            }
            times[index].count += 1.f;
            times[index].gpu += milliseconds;
            times[index].cpu += microseconds;

            // std::cout << stamps[i]
            //           << ": gpu=" << TimeStr(milliseconds * 1000.f)
            //           << ": cpu=" << TimeStr(microseconds) << std::endl;
        }
        // cudaEventElapsedTime(&milliseconds, events.front(), events.back());
        // microseconds = (float)std::chrono::duration_cast<std::chrono::microseconds>(cpu.back() - cpu.front()).count();
        // std::cout
        //     << "GPU Total: " << TimeStr(milliseconds * 1000.f) << std::endl
        //     << "CPU Total: " << TimeStr(microseconds) << std::endl;

        for (auto &time : times)
        {
            time.gpu /= time.count;
            time.cpu /= time.count;
            std::cout << time.name
                      << ": gpu=" << TimeStr(time.gpu * 1000.f)
                      << ": cpu=" << TimeStr(time.cpu) << std::endl;
        }

        for (auto &event : events)
            cudaEventDestroy(event);
    }

    void Stamp(const std::string &name, bool ignore = false)
    {
        if (ignore)
            return;
        auto &event = NextEvent();
        stamps.push_back(name);
        // sync();
        cudaEventRecord(event, 0);
        cpu.push_back(std::chrono::high_resolution_clock::now());
    }
    cudaEvent_t &NextEvent()
    {
        auto index = stamps.size();
        if (index >= events.size())
        {
            auto begin = events.size();
            events.resize(index + 10);
            for (auto i = begin; i < events.size(); ++i)
                cudaEventCreate(&events[i]);
        }
        return events[index];
    }
    std::string TimeStr(float microseconds) const
    {
        std::stringstream ss;
        if (microseconds < 1000.f)
            ss << microseconds << "us";
        else if (microseconds < 1000000.f)
            ss << std::fixed << std::setprecision(2) << microseconds / 1000.f << "MS";
        else
            ss << std::fixed << std::setprecision(2) << microseconds / 1000000.f << "sec";
        return ss.str();
    }
};
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(cu);
