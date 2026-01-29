// relu.h
#ifndef RELU_H
#define RELU_H

#include "Function.h"

class ReLU : public Function {
public:
    [[nodiscard]] Tensor calculate(const Tensor& matrix) override;
    [[nodiscard]] Tensor derivative(const Tensor& matrix, const Tensor& backward) override;
};

#endif // RELU_H
