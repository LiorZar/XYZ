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
using rectangle = glm::vec4;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(ui);

#endif // __GUI_H__