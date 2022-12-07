#include <iostream>

#include "DDouble.hxx"

int main()
{
    DDouble x(3.0f);
    x += 1;
    std::cout << x.as<float>() << std::endl;

    long double y = 1.00000000000000001L;
    DDouble z(y);
    DDouble z2(2);
    z *= z2;
    std::cout << z.hi() << " " << z.lo() << std::endl;

    if (z < std::numeric_limits<DDouble>::epsilon()) {
        std::cout << "Small!";
    }
}
