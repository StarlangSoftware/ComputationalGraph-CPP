// relu.h
#ifndef RELU_H
#define RELU_H

#include "Function.h"

class ReLU : public Function {
public:
    [[nodiscard]] Tensor calculate(const Tensor &value) override;
    [[nodiscard]] Tensor derivative(const Tensor &value, const Tensor &backward) override;
};

#endif // RELU_H
