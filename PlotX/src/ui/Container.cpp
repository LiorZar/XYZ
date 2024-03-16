#include "Container.h"

NAMESPACE_BEGIN(ui);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
REGISTER_NODE_NAMES(Container, "vert", "horz", "none", "container");
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Container::Container(IWnd *wnd, ContainerPtr parent)
    : Control(wnd, parent)
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Container::Parse(const Xml::Node &data)
{
    if (false == Control::Parse(data))
        return false;

    m_gap = data.Get("gap", vec2(0, 0));
    m_enabledChildren = data.Get("enableChildren", true);
    m_layout = ToLayout(data.attr("layout", data.Type()));

    for (auto child = data.First(); child; child = child->Next())
    {
        auto node = m_wnd->GetLibraryNode(child);
        if (nullptr == node)
            continue;

        CreateControl(m_wnd, ME<Container>(), *node);
    }

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Container::PostInit()
{
    if (false == Control::PostInit())
        return false;
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Container::Update(float ts)
{
    if (false == Control::Update(ts))
        return false;

    for (auto &child : m_children)
    {
        if (false == child->Update(ts))
            return false;
    }
    RefreshLayout();

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Container::Draw(float ts) const
{
    if (false == Control::Draw(ts))
        return false;

    for (auto &child : m_children)
    {
        if (false == child->Draw(ts))
            continue;
        child->DrawBorder(ts);
    }

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Container::AddChild(ControlPtr child, int index)
{
    if (nullptr == child)
        return false;

    if (index < 0 || index >= m_children.size())
        m_children.push_back(child);
    else
        m_children.insert(m_children.begin() + index, child);

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Container::RemoveChild(int index)
{
    if (index < 0 || index >= m_children.size())
        return false;

    m_children.erase(m_children.begin() + index);
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Container::RemoveChild(ControlPtr child)
{
    auto it = std::find(m_children.begin(), m_children.end(), child);
    if (it == m_children.end())
        return false;

    m_children.erase(it);
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Container::RemoveAllChildren()
{
    m_children.clear();
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
ControlPtr Container::GetChild(int index) const
{
    if (index < 0 || index >= m_children.size())
        return nullptr;

    return m_children[index];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
ControlPtr Container::GetChild(const std::string &name) const
{
    for (auto &child : m_children)
    {
        if (child->Name() == name)
            return child;
    }
    return nullptr;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Container::CallChildren(const ControlFn &_fn)
{
    for (auto &child : m_children)
        _fn(child);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Container::RefreshSize()
{
    Control::RefreshSize();
    RefreshChildrenSize();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Container::RefreshRectangles()
{
    Control::RefreshRectangles();
    for (auto &child : m_children)
        child->RefreshRectangles();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Container::RefreshChildrenSize()
{
    for (auto &child : m_children)
        child->RefreshSize();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Container::RefreshLayout()
{
    switch (m_layout)
    {
    case eLayout::None:
        refreshLayoutNone();
        break;
    case eLayout::Vertical:
        refreshLayoutVertical();
        break;
    case eLayout::Horizontal:
        refreshLayoutHorizontal();
        break;
    default:
        break;
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Container::OnKey(int key, int scancode, int action, int mods)
{
    if (false == m_enabledChildren)
        return false;

    for (auto &child : m_children)
    {
        if (false == child->OnKey(key, scancode, action, mods))
            return false;
    }
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Container::OnChar(unsigned int codepoint)
{
    if (false == m_enabledChildren)
        return false;

    for (auto &child : m_children)
    {
        if (false == child->OnChar(codepoint))
            return false;
    }
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Container::OnMouse(const vec2 &_point, const eMouseEventType &_event, int _buttons, int _keys, int _wheel)
{
    if (false == Control::OnMouse(_point, _event, _buttons, _keys, _wheel) && _event < eMouseEventType::Out)
        return false;
    if (false == m_enabledChildren)
        return false;

    if (OnMouseChildren(_point, _event, _buttons, _keys, _wheel))
        return true;
    return false;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Container::OnMouseChildren(const vec2 &_point, const eMouseEventType &_event, int _buttons, int _keys, int _wheel)
{
    for (auto it = m_children.rbegin(); it != m_children.rend(); ++it)
    {
        auto &child = *it;
        if (child->OnMouse(_point, _event, _buttons, _keys, _wheel))
            return true;
    }

    return false;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Container::refreshLayoutNone()
{
    vec2 pos;
    for (auto &child : m_children)
    {
        if (false == child->IsVisibleOnly())
            continue;

        const auto &rect = child->GetRectangle();
        pos.x = CalcChildXPosition(m_rectPX, rect, child->GetHAlign(), m_padding, 0);
        pos.y = CalcChildYPosition(m_rectPX, rect, child->GetVAlign(), m_padding, 0);
        child->SetPosition(pos);
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Container::refreshLayoutVertical()
{
    float Y = m_padding.y;
    const auto e = rectangle(0, 0, 0, 0);
    const auto &hAlign = GetHAlign();
    const auto &vAlign = GetVAlign();

    auto pos = vec2(0, 0);
    for (auto &child : m_children)
    {
        if (false == child->IsVisibleOnly())
            continue;

        const auto &rect = child->GetRectangle();
        if (false == child->IIL())
        {
            pos.x = CalcChildXPosition(m_rectPX, rect, child->GetHAlign(), e, 0);
            pos.y = CalcChildYPosition(m_rectPX, rect, child->GetVAlign(), e, 0);
            child->SetPosition(pos);
        }
        else
        {
            pos.x = CalcChildXPosition(m_rectPX, child->GetRectangle(), hAlign, m_padding, 0);
            pos.y = CalcChildYPosition(m_rectPX, child->GetRectangle(), vAlign, e, Y);
            child->SetPosition(pos);
            Y += rect.Height() + m_gap.y;
        }
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Container::refreshLayoutHorizontal()
{
    float X = m_padding.x;
    const auto e = rectangle(0, 0, 0, 0);
    const auto &hAlign = GetHAlign();
    const auto &vAlign = GetVAlign();

    auto pos = vec2(0, 0);
    for (auto &child : m_children)
    {
        if (false == child->IsVisibleOnly())
            continue;

        const auto &rect = child->GetRectangle();
        if (false == child->IIL())
        {
            pos.x = CalcChildXPosition(m_rectPX, rect, child->GetHAlign(), e, 0);
            pos.y = CalcChildYPosition(m_rectPX, rect, child->GetVAlign(), e, 0);
            child->SetPosition(pos);
        }
        else
        {
            pos.x = CalcChildXPosition(m_rectPX, child->GetRectangle(), hAlign, e, X);
            pos.y = CalcChildYPosition(m_rectPX, child->GetRectangle(), vAlign, m_padding, 0);
            child->SetPosition(pos);
            X += rect.Width() + m_gap.x;
        }
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(ui);
