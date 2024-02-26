#include "cu/GPU.h"
#include "SP.cuh"

using namespace cu;
int main()
{
    std::cout << "Current Dir: " << GPU::GetCurrentDirectory() << std::endl;
    std::cout << "Working Dir: " << GPU::GetWorkingDirectory() << std::endl;

    GPU::Set();
    {
        SP sp;
        sp.ToCSV();
        if (false == sp.Init())
            return 1;
        if (false == sp.Process())
            return 2;
        if (false == sp.STFT())
            return 3;
    }

    return 0;
}
