#ifndef __WND_H__
#define __WND_H__

#include "defines.h"

NAMESPACE_BEGIN(ui);

class Wnd
{
private:
    GLFWwindow *window;
    int width, height;
    std::string title;

public:
    Wnd(int width, int height, const char *title);
    ~Wnd();

    std::thread Run();

protected:
    void Loop();

protected:
    virtual bool DrawScene();
    virtual void OnWindowClose();
    virtual void OnWindowSize(int width, int height);
    virtual void OnWindowRefresh();
    virtual void OnMouse(int button, int action, int mods);
    virtual void OnCursorPosition(double xpos, double ypos);
    virtual void OnScroll(double xoffset, double yoffset);
    virtual void OnKey(int key, int scancode, int action, int mods);
    virtual void OnChar(unsigned int codepoint);

private:
    bool shouldClose() const;
    void swapBuffers();
    void pollEvents();
    GLFWwindow *get() const;
};
NAMESPACE_END(ui);

#endif // __WND_H__