#ifndef RELU_HPP
#define RELU_HPP
#include "Function.h"
#include "Matrix.h"

class ReLU : public Function {
public:
    Matrix calculate(const Matrix& matrix) const override;
    Matrix derivative(const Matrix& matrix) const override;
};
#endif // RELU_HPP
