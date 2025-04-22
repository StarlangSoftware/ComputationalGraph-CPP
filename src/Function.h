//
// Created by Yiğit Demirşan on 11.03.2025.
//
#ifndef FUNCTION_H
#define FUNCTION_H

#include "Matrix.h"

class Function {
public:
    virtual ~Function() = default;

    virtual Matrix calculate(const Matrix& matrix) const = 0;
    virtual Matrix derivative(const Matrix& matrix) const = 0;
};

#endif // FUNCTION_H
