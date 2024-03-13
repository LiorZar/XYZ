#include "IWnd.h"

NAMESPACE_BEGIN(ui);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
ControlPtr CreateControl(IWnd *wnd, ContainerPtr parent, const Xml::Node &data)
{
    auto type = data.Type();
    auto fn = NodeFactory::Create(type);
    if (nullptr == fn)
        return nullptr;
    return fn(wnd, parent, data);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Xml::NodePtr IWnd::GetLibraryNode(Xml::NodePtr node) const
{
    return node;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(ui);
