#include <iostream>

#include "DDouble.hxx"

int main()
{
    DDouble x(3.0f);
    x += 1;
    std::cout << (float) x << std::endl;

    long double y = 1.00000'00000'00000'01L;
    DDouble z(y);
    z *= z;
    std::cout << z.hi() << " " << z.lo() << std::endl;
}
