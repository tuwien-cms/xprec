#include <iostream>

#include "DDouble.hxx"

int main()
{
    DDouble x(3.0f);
    x += 1;
    std::cout << (float) x << std::endl;

    long double y = 1.00000000000000001L;
    DDouble z(y);
    z *= z;
    std::cout << z.hi() << " " << z.lo() << std::endl;

    if (z < std::numeric_limits<DDouble>::epsilon()) {
        std::cout << "Small!";
    }
}
