#ifndef TANH_H
#define TANH_H
#include <Matrix.h>
#include "Function.h"

class Tanh : public Function {
public:
    Matrix calculate(const Matrix& matrix) const override;
    Matrix derivative(const Matrix& matrix) const override;
};
#endif // TANH_H
