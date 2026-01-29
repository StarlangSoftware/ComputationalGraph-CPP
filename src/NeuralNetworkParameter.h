//
// Created by Olcay YILDIZ on 28.01.2026.
//

#ifndef COMPUTATIONALGRAPH_NEURALNETWORKPARAMETER_H
#define COMPUTATIONALGRAPH_NEURALNETWORKPARAMETER_H
#include "Initialization/Initialization.h"
#include "Initialization/RandomInitialization.h"
#include "Optimizer/Optimizer.h"


class NeuralNetworkParameter {
private:
    Optimizer* optimizer;
    int epoch;
    int seed;
    Initialization initialization;
    double dropout;
public:
    NeuralNetworkParameter(int seed, int epoch, Optimizer* optimizer, Initialization initialization = RandomInitialization(), double dropout = 0.0);
    [[nodiscard]] Optimizer* getOptimizer() const;
    [[nodiscard]] int getEpoch() const;
    [[nodiscard]] Initialization getInitialization() const;
    [[nodiscard]] double getDropout() const;
};


#endif //COMPUTATIONALGRAPH_NEURALNETWORKPARAMETER_H
