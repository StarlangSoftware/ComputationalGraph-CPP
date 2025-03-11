#pragma once

#include "Function.hpp"
#include <math/Matrix.hpp>

class Tanh : public Function {
public:
    Matrix calculate(const Matrix& matrix) const override;
    Matrix derivative(const Matrix& matrix) const override;
};
