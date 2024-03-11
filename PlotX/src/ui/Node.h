#ifndef __NODE_H__
#define __NODE_H__

#include "IWnd.h"

NAMESPACE_BEGIN(ui);

class Node : public std::enable_shared_from_this<Node>
{
public:
    Node(IWnd *wnd, ContainerPtr parent);
    virtual ~Node() = default;

public:
    virtual bool Parse(const Xml::Node &data);
    virtual bool PostInit();

    // drawing
public:
    virtual bool Update(float ts);
    virtual bool Draw(float ts) const;
    virtual bool DrawBorder(float ts) const;

    // visibility
public:
    virtual void Hide() { m_visible = false; }
    virtual void Show() { m_visible = true; }
    virtual void HideDraw() { m_visibleDraw = false; }
    virtual void ShowDraw() { m_visibleDraw = true; }
    virtual bool IsVisible() const { return m_visible && m_visibleDraw; }
    virtual bool IsVisibleOnly() const { return m_visible; }
    virtual bool IsVisibleDraw() const { return m_visibleDraw; }

    // size & layout
public:
    virtual void RefreshSize(bool _withMargin = true);
    virtual void RefreshRectangles();
    virtual rectangle GetParentRectangle() const;
    virtual rectangle GetParentViewRectangle() const;
    virtual rectangle GetParentViewPort() const;

    bool IIL() const { return m_includeInLayout; }
    eHorzAlign GetHAlign() const { return m_hAlign; }
    eVertAlign GetVAlign() const { return m_vAlign; }
    void SetPosition(const glm::vec2 &_tl);

    // mouse & state
public:
    // virtual bool OnKey(UINT _msg, WPARAM _wParam, int _keys) { return false; }
    virtual bool OnMouse(const glm::vec2 &_point, const eMouseEventType &_event, int _buttons, int _keys, int _wheel);

    virtual bool IsEnable() const { return m_isEnabledMouseEvents; };
    virtual void Enable() { m_isEnabledMouseEvents = true; };
    virtual void Disable() { m_isEnabledMouseEvents = false; };
    virtual void Enable(eBool state)
    {
        if (eBool::False == state)
            Disable();
        else
            Enable();
    }
    virtual bool IsChecked() const { return m_checked; };
    virtual void SetChecked(bool _v)
    {
        m_checked = _v;
        m_state = eNodeState::Normal;
    };
    virtual eNodeState GetState() const { return m_state; }
    virtual std::string GetStateString() const;

    virtual glm::vec2 LocalToGlobal(const glm::vec2 &_pt) const;
    virtual glm::vec2 GlobalToLocal(const glm::vec2 &_pt) const;
    virtual glm::vec2 GlobalToLogicView(const glm::vec2 &_pt) const;

    // tree
public:
    virtual ContainerPtr GetParent() const { return m_parent.lock(); }
    virtual void SetParent(ContainerPtr _control);
    virtual void RemoveFromParent();
    const std::string &Name() const { return m_name; }

    // tree
protected:
    IWnd *m_wnd;
    std::string m_name;
    WeakContainerPtr m_parent;

    // gui
protected:
    eHorzAlign m_hAlign = eHorzAlign::Right;
    eVertAlign m_vAlign = eVertAlign::Top;
    rectangle m_rect;
    rectangle m_padding;

    bool m_visible = false;
    bool m_visibleDraw = false;
    bool m_includeInLayout = true;

    // mouse & state
protected:
    bool m_checked = false;
    bool m_isEnabledMouseEvents = false;
    eNodeState m_state = eNodeState::Normal;
    eNodeState m_disabledState = eNodeState::Normal;
};

NAMESPACE_END(ui);

#endif // __NODE_H__