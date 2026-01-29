#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "Function.h"

class Softmax : public Function {
public:
    [[nodiscard]] Tensor calculate(const Tensor &matrix) override;
    [[nodiscard]] Tensor derivative(const Tensor &value, const Tensor &backward) override;
};
#endif // SOFTMAX_H
