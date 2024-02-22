#include "gl/Wnd.h"

int main()
{
    Wnd window(800, 600, "GLFW Hello World");
    window.Loop();

    return 0;
}
