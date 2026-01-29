#ifndef SIGMOID_H
#define SIGMOID_H
#include "Function.h"

class Sigmoid : public Function {
public:
    [[nodiscard]] Tensor calculate(const Tensor &value) override;
    [[nodiscard]] Tensor derivative(const Tensor &value, const Tensor &backward) override;
};
#endif // SIGMOID_H
