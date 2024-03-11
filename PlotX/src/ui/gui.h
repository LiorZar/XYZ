#ifndef __GUI_H__
#define __GUI_H__

#include "defines.h"
#include "vec.h"

NAMESPACE_BEGIN(ui);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
enum class eHorzAlign
{
    Left,
    Center,
    Right
};
enum class eVertAlign
{
    Top,
    Middle,
    Bottom
};
enum class eLayout
{
    None,
    Vertical,
    Horizontal
};
enum class eNodeState
{
    Normal,
    Hover,
    Down,
    Disabled,
    Hidden
};
enum class eMouseEventType
{
    None,
    Down,
    Up,
    Move,
    DoubleClick,
    Wheel,
    Out,
    Leave
};
enum class eBool
{
    False,
    True,
    DontCare
};
struct eMouseButtonType
{
    enum
    {
        None = 0,
        Left = 1,
        Right = 2,
        Middle = 4
    };
};
struct eKeysStateType
{
    enum
    {
        None = 0,
        Control = 1,
        Shift = 2,
        Alt = 4
    };
};
//-------------------------------------------------------------------------------------------------------------------------------------------------//
eHorzAlign ToHorzAlign(const std::string &str);
eVertAlign ToVertAlign(const std::string &str);
eLayout ToLayout(const std::string &str);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
using color = vec4;
struct rectangle : public vec4
{
    rectangle(float x = 0, float y = 0, float z = 0, float w = 0) : vec4(x, y, z, w) {}
    rectangle(const vec2 &tl, const vec2 &br) : vec4(tl.x, tl.y, br.x, br.y) {}
    rectangle(const vec4 &r) : vec4(r) {}

    float Width() const { return z - x; }
    float Height() const { return w - y; }
    vec2 Size() const { return vec2(Width(), Height()); }
    vec2 TL() const { return vec2(x, y); }
    vec2 BR() const { return vec2(z, w); }
    vec2 Center() const { return vec2((x + z) / 2, (y + w) / 2); }
    void TL(const vec2 &tl)
    {
        auto s = Size();
        x = tl.x;
        y = tl.y;
        z = x + s.x;
        w = y + s.y;
    }
    void BR(const vec2 &br)
    {
        auto s = Size();
        z = br.x;
        w = br.y;
        x = z - s.x;
        y = w - s.y;
    }
    bool IsPointInside(const vec2 &pt) const
    {
        return pt.x >= x && pt.x <= z && pt.y >= y && pt.y <= w;
    }
};
//-------------------------------------------------------------------------------------------------------------------------------------------------//
static inline bool IsBitOn(int _bits, int _value) { return 0 != (_bits & _value); }
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(ui);

#endif // __GUI_H__