#pragma once

#include "Function.hpp"
#include <math/Matrix.hpp>
#include <cmath>

class Sigmoid : public Function {
public:
    Matrix calculate(const Matrix& matrix) const override;
    Matrix derivative(const Matrix& matrix) const override;
};
