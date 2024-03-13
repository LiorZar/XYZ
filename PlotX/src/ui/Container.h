#ifndef ___CONTAINER_H___
#define ___CONTAINER_H___

#include "Control.h"

NAMESPACE_BEGIN(ui);

class Container : public Control
{
public:
    Container(IWnd *wnd, ContainerPtr parent);
    virtual ~Container() = default;

public:
    virtual bool Parse(const Xml::Node &data);
    virtual bool PostInit();

    // drawing
public:
    virtual bool Update(float ts) override;
    virtual bool Draw(float ts) const override;

    // children
public:
    virtual bool AddChild(ControlPtr child, int index = -1);
    virtual bool RemoveChild(int index);
    virtual bool RemoveChild(ControlPtr child);
    virtual bool RemoveAllChildren();
    virtual ControlPtr GetChild(int index) const;
    virtual ControlPtr GetChild(const std::string &name) const;
    virtual int GetChildCount() const { return m_children.size(); }
    virtual void CallChildren(const ControlFn &_fn);

    // size & layout
public:
    virtual void RefreshSize() override;
    virtual void RefreshRectangles() override;
    virtual void RefreshChildrenSize();
    virtual void RefreshLayout();

    // mouse & state
public:
    virtual bool OnKey(int key, int scancode, int action, int mods) override;
    virtual bool OnChar(unsigned int codepoint) override;
    virtual bool OnMouse(const vec2 &_point, const eMouseEventType &_event, int _buttons, int _keys, int _wheel) override;
    virtual bool OnMouseChildren(const vec2 &_point, const eMouseEventType &_event, int _buttons, int _keys, int _wheel);

    virtual void SetEnableChildren(bool _v) { m_enabledChildren = _v; }

    // virtual vec2 LocalToGlobal(const vec2 &_pt) const override;
    // virtual vec2 GlobalToLocal(const vec2 &_pt) const override;

protected:
    virtual void refreshLayoutNone();
    virtual void refreshLayoutVertical();
    virtual void refreshLayoutHorizontal();

protected:
    vec2 m_gap;
    bool m_enabledChildren = true;
    eLayout m_layout = eLayout::None;
    std::vector<ControlPtr> m_children;
};

NAMESPACE_END(ui);

#endif // ___CONTAINER_H___
