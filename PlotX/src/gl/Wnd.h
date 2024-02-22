#ifndef __WND_H__
#define __WND_H__

#include "defines.h"

NAMESPACE_BEGIN(gl);

class Wnd
{
private:
    GLFWwindow *window;
    int width, height;

public:
    Wnd(int width, int height, const char *title);
    ~Wnd();

    bool Loop();

protected:
    virtual bool DrawScene();
    virtual void OnMouse(int button, int action, int mods);
    virtual void OnKey(int key, int scancode, int action, int mods);
    virtual void OnWindowClose();
    virtual void OnWindowSize(int width, int height);
    virtual void OnWindowRefresh();
    virtual void OnCursorPosition(double xpos, double ypos);
    virtual void OnScroll(double xoffset, double yoffset);
    virtual void OnChar(unsigned int codepoint);

private:
    bool shouldClose() const;
    void swapBuffers();
    void pollEvents();
    GLFWwindow *get() const;
};
NAMESPACE_END(gl);

#endif // __WND_H__