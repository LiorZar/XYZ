#include "cu/GPU.h"
#include "SP.cuh"

using namespace cu;
int main()
{
	std::cout << "Current Dir: " << GPU::GetCurrentDirectory() << std::endl;
	std::cout << "Working Dir: " << GPU::GetWorkingDirectory() << std::endl;

	{
		SP sp;

		sp.Init();
	}
    
    return 0;
}
