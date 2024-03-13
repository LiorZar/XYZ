#ifndef __NODE_H__
#define __NODE_H__

#include "IWnd.h"

NAMESPACE_BEGIN(ui);

class Control : public std::enable_shared_from_this<Control>
{
public:
    Control(IWnd *wnd, ContainerPtr parent);
    virtual ~Control() = default;

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
    virtual void RefreshSize();
    virtual void RefreshRectangles();
    virtual rectangle GetRectangle() const { return m_rectPX; }
    virtual rectangle GetParentRectangle() const;

    bool IIL() const { return m_includeInLayout; }
    eHorzAlign GetHAlign() const { return m_hAlign; }
    eVertAlign GetVAlign() const { return m_vAlign; }
    void SetPosition(const vec2 &_tl);

    // mouse & state
public:
    virtual bool OnKey(int key, int scancode, int action, int mods) { return false; }
    virtual bool OnChar(unsigned int codepoint) { return false; }
    virtual bool OnMouse(const vec2 &_point, const eMouseEventType &_event, int _buttons, int _keys, int _wheel);

    virtual bool IsEnable() const { return m_enabled; };
    virtual void Enable() { m_enabled = true; };
    virtual void Disable() { m_enabled = false; };
    virtual void Enable(eBool state);
    virtual bool IsChecked() const { return m_checked; };
    virtual void SetChecked(bool _v);
    virtual eNodeState GetState() const { return m_state; }
    virtual std::string GetStateString() const;

    // virtual vec2 LocalToGlobal(const vec2 &_pt) const;
    // virtual vec2 GlobalToLocal(const vec2 &_pt) const;

    // tree
public:
    template <typename T>
    std::shared_ptr<T> ME() { return std::dynamic_pointer_cast<T>(shared_from_this()); }
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
    rectangle m_rect = {0, 0, 1, 1};
    rectangle m_rectPX = {0, 0, 0, 0};
    rectangle m_padding = {0, 0, 0, 0};
    color m_color = {1, 1, 1, 1};

    bool m_visible = true;
    bool m_visibleDraw = true;
    bool m_visibleBorder = false;
    bool m_includeInLayout = true;

    // mouse & state
protected:
    bool m_checked = false;
    bool m_enabled = false;
    eNodeState m_state = eNodeState::Normal;

protected:
    mat3 m_transform = mat3(1.f);
    mat3 m_globalTransform = mat3(1.f);
};

NAMESPACE_END(ui);

#endif // __NODE_H__