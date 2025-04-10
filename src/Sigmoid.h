#ifndef SIGMOID_H
#define SIGMOID_H
#include "Function.h"
#include <Matrix.h>

class Sigmoid : public Function {
public:
    Matrix calculate(const Matrix& matrix) const override;
    Matrix derivative(const Matrix& matrix) const override;
};
#endif // SIGMOID_H
