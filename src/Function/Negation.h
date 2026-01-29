//
// Created by Olcay YILDIZ on 26.01.2026.
//

#ifndef COMPUTATIONALGRAPH_NEGATION_H
#define COMPUTATIONALGRAPH_NEGATION_H
#include "Function.h"

class Negation : public Function {
public:
    [[nodiscard]] Tensor calculate(const Tensor &value) override;
    [[nodiscard]] Tensor derivative(const Tensor &value, const Tensor &backward) override;
};


#endif //COMPUTATIONALGRAPH_NEGATION_H