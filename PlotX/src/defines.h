#ifndef __DEFINES_H__
#define __DEFINES_H__

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdint>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <list>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <memory>
#include <thread>
#include <functional>
#include <chrono>
#include <stdexcept>

#define NAMESPACE_BEGIN(name) \
    namespace name            \
    {
#define NAMESPACE_END(name) }

using bul = bool;
using s08 = std::int8_t;
using u08 = std::uint8_t;
using s16 = std::int16_t;
using u16 = std::uint16_t;
using s32 = std::int32_t;
using u32 = std::uint32_t;
using s64 = std::int64_t;
using u64 = std::uint64_t;
using f32 = float;
using f64 = double;

#endif // __DEFINES_H__