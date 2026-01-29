//
// Created by Yiğit Demirşan on 11.03.2025.
//
#ifndef FUNCTION_H
#define FUNCTION_H

#include <Tensor.h>

#include "FunctionType.h"

class Function {
public:
    FunctionType type = FunctionType::DEFAULT;
    virtual ~Function() = default;
    [[nodiscard]] virtual Tensor calculate(const Tensor& matrix) = 0;
    [[nodiscard]] virtual Tensor derivative(const Tensor& value, const Tensor& backward) = 0;
};

#endif // FUNCTION_H
