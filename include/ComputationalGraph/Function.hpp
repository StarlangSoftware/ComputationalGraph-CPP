//
// Created by Yiğit Demirşan on 11.03.2025.
//

#pragma once

#include <math/Matrix.hpp>

class Function {
public:
    virtual ~Function() = default;

    virtual Matrix calculate(const Matrix& matrix) const = 0;
    virtual Matrix derivative(const Matrix& matrix) const = 0;
};
