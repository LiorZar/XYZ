#include "defines.h"

NAMESPACE_BEGIN(cu);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
class Elapse
{
private:
    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
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
        auto &event = Stamp("End");
        cudaEventSynchronize(event);

        float milliseconds = 0, microseconds = 0;
        for (auto i = 1U; i < stamps.size(); ++i)
        {
            auto &start = events[i - 1];
            auto &end = events[i];
            cudaEventElapsedTime(&milliseconds, start, end);
            microseconds = (float)std::chrono::duration_cast<std::chrono::microseconds>(cpu[i] - cpu[i - 1]).count();

            std::cout << stamps[i]
                      << ": gpu=" << TimeStr(milliseconds * 1000.f)
                      << ": cpu=" << TimeStr(microseconds) << std::endl;
        }
        cudaEventElapsedTime(&milliseconds, events[0], events[events.size() - 1]);
        std::cout
            << "GPU Total: " << TimeStr(milliseconds * 1000.f) << std::endl
            << "CPU Total: " << TimeStr(microseconds) << std::endl;

        for (auto &event : events)
            cudaEventDestroy(event);
    }

    cudaEvent_t &Stamp(const std::string &name)
    {
        auto &event = NextEvent();
        stamps.push_back(name);
        sync();
        cudaEventRecord(event);
		cpu.push_back(std::chrono::high_resolution_clock::now());
		return event;
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
