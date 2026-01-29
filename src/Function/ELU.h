//
// Created by Olcay YILDIZ on 26.01.2026.
//

#ifndef COMPUTATIONALGRAPH_ELU_H
#define COMPUTATIONALGRAPH_ELU_H
#include "Function.h"

class ELU : public Function {
private:
    double a;
public:
    explicit ELU(double a);
    ELU();
    [[nodiscard]] Tensor calculate(const Tensor &value) override;
    [[nodiscard]] Tensor derivative(const Tensor &value, const Tensor &backward) override;
};


#endif //COMPUTATIONALGRAPH_ELU_H