//
// Created by Olcay YILDIZ on 28.01.2026.
//

#ifndef COMPUTATIONALGRAPH_SGDMOMENTUM_H
#define COMPUTATIONALGRAPH_SGDMOMENTUM_H
#include "Optimizer.h"


class SGDMomentum : public Optimizer{
protected:
    map<ComputationalNode, vector<double>> velocityMap;
    double momentum;
    void setGradients(ComputationalNode &node) override;
public:
    SGDMomentum(double learningRate, double etaDecrease, double momentum);
};


#endif //COMPUTATIONALGRAPH_SGDMOMENTUM_H