// relu.h
#ifndef RELU_H
#define RELU_H

#include "Function.h"
#include "Matrix.h"

class ReLU : public Function {
public:
    Matrix calculate(const Matrix& matrix) const override;
    Matrix derivative(const Matrix& matrix) const override;
};

#endif // RELU_H
