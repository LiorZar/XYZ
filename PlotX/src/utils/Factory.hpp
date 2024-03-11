#include "defines.h"
#ifndef __FACTORY_H__
#define __FACTORY_H__

template <typename T, typename FN>
class Factory
{
private:
    Factory() = default;
    Factory(const Factory &) = delete;
    Factory &operator=(const Factory &) = delete;

public:
    static Factory &Get()
    {
        static Factory instance;
        return instance;
    }
    ~Factory() = default;

    static void Register(const std::string &name, FN fn)
    {
        Get().m_map[name] = fn;
    }

    static T Create(const std::string &name)
    {
        auto &mp = Get().m_map;
        auto it = mp.find(name);
        if (it != mp.end())
        {
            return it->second();
        }
        return nullptr;
    }

private:
    std::map<std::string, FN> m_map;
};

#endif // __FACTORY_H__