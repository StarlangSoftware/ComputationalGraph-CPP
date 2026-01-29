//
// Created by Olcay YILDIZ on 26.01.2026.
//

#ifndef COMPUTATIONALGRAPH_DROPOUT_H
#define COMPUTATIONALGRAPH_DROPOUT_H
#include <random>
#include "Function.h"

class Dropout : public Function {
private:
    double p;
    default_random_engine randomEngine;
    vector<double> mask;
public:
    Dropout(double p, const default_random_engine &randomEngine);
    [[nodiscard]] Tensor calculate(const Tensor &value) override;
    [[nodiscard]] Tensor derivative(const Tensor &value, const Tensor &backward) override;
};


#endif //COMPUTATIONALGRAPH_DROPOUT_H