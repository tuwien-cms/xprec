#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

/**
 * Class for double-double arithmetic.
 *
 * Emulates quadruple precision with a pair of doubles.  This roughly doubles
 * the mantissa bits (and thus squares the precision of double).  The range
 * is almost the same as double, with a larger area of denormalized numbers.
 */
class DDouble {
public:
    constexpr DDouble(double x=0) : _hi(x), _lo(0.0) { }
    constexpr DDouble(long double x) : _hi(x), _lo(x - _hi) { }

    constexpr DDouble(std::int64_t x) : _hi(x), _lo(x - _hi) { }
    constexpr DDouble(std::uint64_t x) : _hi(x), _lo(x - _hi) { }
    constexpr DDouble(std::int32_t x) : _hi(x), _lo(0.0) { }
    constexpr DDouble(std::uint32_t x) : _hi(x), _lo(0.0) { }

    /** Perform ddouble-accurate sum of two doubles */
    static DDouble sum(double a, double b);

    /** Perform ddouble-accurate product of two doubles */
    static DDouble product(double a, double b);

    /** Get high part of the ddouble */
    constexpr double hi() const { return _hi; }

    /** Get low part of the ddouble */
    constexpr double lo() const { return _lo; }

    /** Convert ddouble to different type */
    template <typename T>
    constexpr T as();

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

protected:
    constexpr DDouble(double hi, double lo) : _hi(hi), _lo(lo) { }

    static DDouble fast_sum(double a, double b);

private:
    double _hi;
    double _lo;

    friend class std::numeric_limits<DDouble>;
};

namespace std {

/**
 * Specialization of numerical limits for the double-double type.
 */
template <>
class numeric_limits<DDouble> {
    using _double = numeric_limits<double>;

public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;

    static constexpr bool has_infinity = _double::has_infinity;
    static constexpr bool has_quiet_NaN = _double::has_quiet_NaN;
    static constexpr bool has_signaling_NaN = _double::has_signaling_NaN;
    static constexpr float_denorm_style has_denorm = _double::has_denorm;
    static constexpr bool has_denorm_loss = false;

    static constexpr float_round_style round_style = _double::round_style;
    static constexpr int digits = 2 * _double::digits + 1;
    static constexpr int digits10 = 2 * _double::digits10;
    static constexpr int max_digits10 = 2 * _double::max_digits10;

    static constexpr int radix = _double::radix;

    static constexpr int min_exponent = _double::min_exponent + _double::digits;
    static constexpr int min_exponent10 = _double::min_exponent10 + _double::digits10;
    static constexpr int max_exponent = _double::max_exponent;
    static constexpr int max_exponent10 = _double::max_exponent10;

    static constexpr DDouble min() noexcept;
    static constexpr DDouble max() noexcept;
    static constexpr DDouble lowest() noexcept;
    static constexpr DDouble epsilon() noexcept;
    static constexpr DDouble round_error() noexcept;

    static constexpr DDouble infinity() noexcept;
    static constexpr DDouble quiet_NaN() noexcept;
    static constexpr DDouble signaling_NaN() noexcept;
    static constexpr DDouble denorm_min() noexcept;

    static constexpr bool is_bounded = numeric_limits<double>::is_bounded;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_modulo = false;

    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
};

}

// ======================== Implementation ===========================

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

template <>
constexpr long double DDouble::as<long double>()
{
    return (long double) hi() + lo();
}

template <> constexpr double DDouble::as<double>() { return hi(); }
template <> constexpr float DDouble::as<float>() { return hi(); }


namespace std {

constexpr DDouble numeric_limits<DDouble>::min() noexcept
{
    // Whereas the maximum exponent is the same for double and DDouble,
    // Denormalization in the low part means that the min exponent for
    // normalized values is lower.
    return DDouble(_double::min() / _double::epsilon());
}

constexpr DDouble numeric_limits<DDouble>::max() noexcept
{
    return DDouble(_double::max(),
                    _double::max() / _double::epsilon() / _double::radix);
}

constexpr DDouble numeric_limits<DDouble>::lowest() noexcept
{
    return DDouble(_double::lowest(),
                    _double::lowest() / _double::epsilon() / _double::radix);
}

constexpr DDouble numeric_limits<DDouble>::epsilon() noexcept
{
    return DDouble(_double::epsilon() * _double::epsilon() / _double::radix);
}

constexpr DDouble numeric_limits<DDouble>::round_error() noexcept
{
    return DDouble(_double::round_error());
}

constexpr DDouble numeric_limits<DDouble>::infinity() noexcept
{
    return DDouble(_double::infinity(), _double::infinity());
}

constexpr DDouble numeric_limits<DDouble>::quiet_NaN() noexcept
{
    return DDouble(_double::quiet_NaN(), _double::quiet_NaN());
}

constexpr DDouble numeric_limits<DDouble>::signaling_NaN() noexcept
{
    return DDouble(_double::signaling_NaN(), _double::signaling_NaN());
}

constexpr DDouble numeric_limits<DDouble>::denorm_min() noexcept
{
    return DDouble(_double::denorm_min());
}

} /* namespace std */
