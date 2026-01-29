//
// Created by Olcay YILDIZ on 28.01.2026.
//

#ifndef COMPUTATIONALGRAPH_STOCHASTICGRADIENTDESCENT_H
#define COMPUTATIONALGRAPH_STOCHASTICGRADIENTDESCENT_H
#include "Optimizer.h"


class StochasticGradientDescent : public Optimizer{
public:
    StochasticGradientDescent(double learningRate, double etaDecrease);
protected:
    void setGradients(ComputationalNode &node) override;
};


#endif //COMPUTATIONALGRAPH_STOCHASTICGRADIENTDESCENT_H