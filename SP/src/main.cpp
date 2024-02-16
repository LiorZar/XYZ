#include "cl/defines.h"
#include "cl/Program.h"
#include "cl/Buffer.hpp"

int main()
{
    Context::getInstance();
    Program program("../../src/kernels/main.cl");

    Buffer<float> A(1000), B(1000), C(1000);
    for (int i = 0; i < 1000; i++)
    {
        A[i] = i;
        B[i] = i * 2;
    }
    int t = program.Dispatch1D("addVectors", 1000, 256, A.d(), B.d(), C.d(), 1000);
    C.Download();
    for (int i = 0; i < 1000; i++)
    {
        std::cout << C[i] << std::endl;
    }

    return 0;
}
