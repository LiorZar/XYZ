#include "FFT.h"

NAMESPACE_BEGIN(cu);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
size_t FFT::NextPow2(size_t n)
{
    size_t p = 1;
    while (p < n)
        p <<= 1;
    return p;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
size_t FFT::NextPow235(size_t n)
{
    std::vector<size_t> numbers = {1};
    size_t i = 0, j = 0, k = 0; // Indices for 2, 3, and 5 multiples

    while (numbers.back() < n)
    {
        size_t next2 = numbers[i] * 2;
        size_t next3 = numbers[j] * 3;
        size_t next5 = numbers[k] * 5;

        size_t nextNumber = std::min<size_t>({next2, next3, next5});
        numbers.push_back(nextNumber);

        if (nextNumber == next2)
            ++i;
        if (nextNumber == next3)
            ++j;
        if (nextNumber == next5)
            ++k;
    }

    return numbers.back();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//

NAMESPACE_END(cu);