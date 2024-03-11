#include "Node.h"
#include "Container.h"

NAMESPACE_BEGIN(ui);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Node::Node(IWnd *wnd, ContainerPtr parent) : m_wnd(wnd), m_parent(parent)
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::Parse(const Xml::Node &data)
{
    m_name = data.attr("name", "");
    m_hAlign = ToHorzAlign(data.attr("horz", "center"));
    m_vAlign = ToVertAlign(data.attr("vert", "center"));
    m_rect = data.Get("rect", rectangle(0, 0, 1, 1));
    m_padding = data.Get("padding", rectangle(0, 0, 0, 0));
    m_color = data.getColor("color", color(1, 1, 1, 1));
    m_visible = data.Get("visible", true);
    m_visibleDraw = data.Get("visibleDraw", true);
    m_visibleBorder = data.Get("visibleBorder", false);
    m_includeInLayout = data.Get("iil", true);
    m_isEnabledMouseEvents = data.Get("mouseEnabled", true);

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::PostInit()
{
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::Update(float ts)
{
    RefreshSize();
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::Draw(float ts) const
{
    return IsVisible();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::DrawBorder(float ts) const
{
    if (false == IsVisible() || false == m_visibleBorder)
        return false;

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Node::RefreshSize(bool _withMargin)
{
    auto prect = GetParentRectangle();
    auto size = prect.Size();
    m_rectPX.x = m_rect.x * size.x + m_padding.x;
    m_rectPX.y = m_rect.y * size.y + m_padding.y;
    m_rectPX.z = m_rect.z * size.x - m_padding.z;
    m_rectPX.w = m_rect.w * size.y - m_padding.w;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Node::RefreshRectangles()
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
rectangle Node::GetParentRectangle() const
{
    auto parent = m_parent.lock();
    if (nullptr != parent)
        return rectangle(0, 0, 0, 0);
    return parent->GetRectangle();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::OnMouse(const vec2 &_point, const eMouseEventType &_event, int _buttons, int _keys, int _wheel)
{
    m_state = eNodeState::Normal;
    if (false == IsVisible())
    {
        m_state = eNodeState::Hidden;
        return false;
    }
    if (false == IsEnable())
    {
        m_state = eNodeState::Disabled;
        return false;
    }
    bool isInside = m_rectPX.IsPointInside(_point);
    if (isInside)
        m_state = IsBitOn(_buttons, eMouseButtonType::Left) ? eNodeState::Down : eNodeState::Hover;

    return isInside;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Node::Enable(eBool state)
{
    if (eBool::False == state)
        Disable();
    else
        Enable();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Node::SetChecked(bool _v)
{
    m_checked = _v;
    m_state = eNodeState::Normal;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
std::string Node::GetStateString() const
{
    std::string state = "";
    switch (m_state)
    {
    case eNodeState::Hover:
        state = "Hover";
        break;
    case eNodeState::Down:
        state = "Down";
        break;
    case eNodeState::Disabled:
        state = "Disabled";
        break;
    case eNodeState::Hidden:
        state = "Hidden";
        break;
    }
    if (m_checked)
        state += "_Checked";
    return state;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Node::SetParent(ContainerPtr _control)
{
    if (_control == GetParent())
        return;
    RemoveFromParent();
    m_parent = _control;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Node::RemoveFromParent()
{
    auto parent = m_parent.lock();
    // if (nullptr != parent)
    //     parent->RemoveChild(shared_from_this());
    m_parent.reset();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(ui);