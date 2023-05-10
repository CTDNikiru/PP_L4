#include <cmath>

using namespace std;

class MathFunc
{
public:
    double a = 0;

    MathFunc(double a)
    {
        this->a = a;
    }

    double phi(double x, double y, double z)
    {
        return pow(x, 2) + pow(y, 2) + pow(z, 2);
    }

    double rho(double phiValue)
    {
        return 6.0 - a * phiValue;
    }

    // Перевод из координаты сетки в настоящее значение
    double toReal(double start, double step, int bias)
    {
        return start + step * bias;
    }

    // Модуль отклонения
    double getDeviation(double gridValue, double xLocalStart, double yStart, double zStart, double hx, double hy, double hz, int i, int j, int k)
    {
        return abs(gridValue - phi(toReal(xLocalStart, hx, i), toReal(yStart, hy, j), toReal(zStart, hz, k)));
    }
};