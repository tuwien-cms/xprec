#pragma once
#include <cstdint>
#include <cmath>

class DDouble {
private:
    double _hi;
    double _lo;

public:
    DDouble() : _hi(), _lo() { }

    DDouble(long double x) : _hi(x), _lo(x - _hi) { }

    DDouble(double x) : _hi(x), _lo(0.0) { }

    double hi() const { return _hi; }

    double lo() const { return _lo; }

    friend DDouble operator+(DDouble x, double y);
    friend DDouble operator+(DDouble x, DDouble y);
    friend DDouble operator+(double x, DDouble y) { return y + x; }

    friend DDouble operator-(DDouble x, double y) { return x + (-y); }
    friend DDouble operator-(double x, DDouble y) { return x + (-y); }
    friend DDouble operator-(DDouble x, DDouble y) { return x + (-y); }
    friend DDouble operator-(DDouble x) { return DDouble(-x.hi(), -x.lo()); }

    friend DDouble operator*(DDouble x, double y);
    friend DDouble operator*(DDouble x, DDouble y);
    friend DDouble operator*(double x, DDouble y) { return y * x; }

    friend DDouble operator/(DDouble x, double y);
    friend DDouble operator/(DDouble x, DDouble y);

    DDouble &operator+=(double y) { return *this = *this + y; }
    DDouble &operator-=(double y) { return *this = *this - y; }
    DDouble &operator*=(double y) { return *this = *this * y; }
    DDouble &operator/=(double y) { return *this = *this / y; }

    DDouble &operator+=(DDouble y) { return *this = *this + y; }
    DDouble &operator-=(DDouble y) { return *this = *this - y; }
    DDouble &operator*=(DDouble y) { return *this = *this * y; }
    DDouble &operator/=(DDouble y) { return *this = *this / y; }

    friend bool operator==(DDouble x, DDouble y);
    friend bool operator!=(DDouble x, DDouble y);
    friend bool operator<=(DDouble x, DDouble y);
    friend bool operator< (DDouble x, DDouble y);
    friend bool operator>=(DDouble x, DDouble y);
    friend bool operator> (DDouble x, DDouble y);

    friend bool operator==(DDouble x, double y) { return x == DDouble(y); }
    friend bool operator!=(DDouble x, double y) { return x != DDouble(y); }
    friend bool operator<=(DDouble x, double y) { return x <= DDouble(y); }
    friend bool operator>=(DDouble x, double y) { return x >= DDouble(y); }
    friend bool operator> (DDouble x, double y) { return x > DDouble(y); }

    friend bool operator==(double x, DDouble y) { return DDouble(x) == y; }
    friend bool operator!=(double x, DDouble y) { return DDouble(x) != y; }
    friend bool operator<=(double x, DDouble y) { return DDouble(x) <= y; }
    friend bool operator>=(double x, DDouble y) { return DDouble(x) >= y; }
    friend bool operator> (double x, DDouble y) { return DDouble(x) > y; }

    operator long double () const { return _to_float<long double>(); }
    operator double () const { return _to_float<double>(); }
    operator float () const { return _to_float<float>(); }

    operator bool () const { return (bool) hi(); }

    static DDouble sum(double a, double b);

    static DDouble product(double a, double b);

protected:
    DDouble(double hi, double lo) : _hi(hi), _lo(lo) { }

    static DDouble fast_sum(double a, double b);

private:
    template <typename T>
    T _to_float() const {
        return sizeof(T) > sizeof(double) ? T(_hi) + _lo : T(_hi);
    }
};

inline DDouble DDouble::fast_sum(double a, double b)
{
    // Algorithm 1: cost 3 flops
    double s = a + b;
    double z = s - a;
    double t = b - z;
    return DDouble(s, t);
}

inline DDouble DDouble::sum(double a, double b)
{
    // Algorithm 2: cost 6 flops
    double s = a + b;
    double aprime = s - b;
    double bprime = s - aprime;
    double delta_a = a - aprime;
    double delta_b = b - bprime;
    double t = delta_a + delta_b;
    return DDouble(s, t);
}

inline DDouble DDouble::product(double a, double b)
{
    // Algorithm 3: cost 2 flops
    double pi = a * b;
    double rho = std::fma(a, b, -pi);
    return DDouble(pi, rho);
}

inline DDouble operator+(DDouble x, double y)
{
    // Algorithm 4: cost 10 flops, error 2 u^2
    DDouble s = DDouble::sum(x.hi(), y);
    double v = x.lo() + s.lo();
    return DDouble::fast_sum(s.hi(), v);
}

inline DDouble operator+(DDouble x, DDouble y)
{
    // Algorithm 6: cost 20 flops, error 3 u^2 + 13 u^3
    DDouble s = DDouble::sum(x.hi(), y.hi());
    DDouble t = DDouble::sum(x.lo(), y.lo());
    double c = s.lo() + t.hi();
    DDouble v = DDouble::fast_sum(s.hi(), c);
    double w = t.lo() + v.lo();
    return DDouble::fast_sum(v.hi(), w);
}

inline DDouble operator*(DDouble x, double y)
{
    // Algorithm 9: cost 6 flops, error 2 u^2
    DDouble c = DDouble::product(x.hi(), y);
    double cl3 = std::fma(x.lo(), y, c.lo());
    return DDouble::fast_sum(c.hi(), cl3);
}

inline DDouble operator*(DDouble x, DDouble y)
{
    // Algorithm 12: cost 9 flops, error 5 u^2
    DDouble c = DDouble::product(x.hi(), y.hi());
    double tl0 = x.lo() * y.lo();
    double tl1 = std::fma(x.hi(), y.lo(), tl0);
    double cl2 = std::fma(x.lo(), y.hi(), tl1);
    double cl3 = c.lo() + cl2;
    return DDouble::fast_sum(c.hi(), cl3);
}

inline DDouble operator/(DDouble x, double y)
{
    // Algorithm 15: cost 10 flops, error 3 u^2
    double th = x.hi() / y;
    DDouble pi = DDouble::product(th, y);
    double delta_h = x.hi() - pi.hi();
    double delta_tee = delta_h - pi.lo();
    double delta = delta_tee + x.lo();
    double tl = delta / y;
    return DDouble::fast_sum(th, tl);
}

inline DDouble operator/(DDouble x, DDouble y)
{
    // Algorithm 18: cost 31 flops, error 10 u^2
    double th = x.hi() / y.hi();
    double rh = 1 - y.hi() * th;
    double rl = -y.lo() * th;
    DDouble e = DDouble::fast_sum(rh, rl);
    DDouble delta = e * th;
    DDouble m = delta + th;
    return x * m;
}

inline bool operator==(DDouble x, DDouble y)
{
    return x.hi() == y.hi() && x.lo() == y.lo();
}

inline bool operator!=(DDouble x, DDouble y)
{
    return x.hi() != y.hi() || x.lo() != y.lo();
}

inline bool operator<=(DDouble x, DDouble y)
{
    return x.hi() < y.hi() || (x.hi() == y.hi() && x.lo() <= y.lo());
}

inline bool operator<(DDouble x, DDouble y)
{
    return x.hi() < y.hi() || (x.hi() == y.hi() && x.lo() < y.lo());
}

inline bool operator>=(DDouble x, DDouble y)
{
    return x.hi() > y.hi() || (x.hi() == y.hi() && x.lo() >= y.lo());
}

inline bool operator>(DDouble x, DDouble y)
{
    return x.hi() > y.hi() || (x.hi() == y.hi() && x.lo() > y.lo());
}
