#ifndef TANH_H
#define TANH_H
#include "Function.h"

class Tanh : public Function {
public:
    [[nodiscard]] Tensor calculate(const Tensor &tensor) override;
    [[nodiscard]] Tensor derivative(const Tensor &value, const Tensor &backward) override;
};
#endif // TANH_H
