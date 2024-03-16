#ifndef __WND_H__
#define __WND_H__

#include "Container.h"

NAMESPACE_BEGIN(ui);

class Wnd : public Container, public IWnd
{
public:
    Wnd(const std::string &workDir);
    ~Wnd();

public:
    bool Load(const std::string &filename);
    std::thread Start();

protected:
    void Run();

protected:
    virtual bool Update();
    virtual bool PreScene();
    virtual bool DrawScene();
    virtual bool PostScene();
    virtual void OnWindowClose();
    virtual void OnWindowSize(int width, int height);
    virtual void OnWindowRefresh();
    virtual void OnMouse(int button, int action, int mods);
    virtual void OnCursorPosition(double xpos, double ypos);
    virtual void OnScroll(double xoffset, double yoffset);
    virtual bool OnKey(int key, int scancode, int action, int mods) override;
    virtual bool OnChar(unsigned int codepoint) override;

private:
    GLFWwindow *window = nullptr;
};
NAMESPACE_END(ui);

#endif // __WND_H__