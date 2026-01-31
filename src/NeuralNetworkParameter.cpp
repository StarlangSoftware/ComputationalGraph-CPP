//
// Created by Olcay YILDIZ on 28.01.2026.
//

#include "NeuralNetworkParameter.h"

NeuralNetworkParameter::NeuralNetworkParameter(const int seed, const int epoch, Optimizer *optimizer,
    Initialization* initialization, const double dropout) {
    this->seed = seed;
    this->epoch = epoch;
    this->optimizer = optimizer;
    this->initialization = initialization;
    this->dropout = dropout;
}

Optimizer * NeuralNetworkParameter::getOptimizer() const {
    return optimizer;
}

int NeuralNetworkParameter::getEpoch() const {
    return epoch;
}

Initialization* NeuralNetworkParameter::getInitialization() const {
    return initialization;
}

double NeuralNetworkParameter::getDropout() const {
    return dropout;
}

int NeuralNetworkParameter::getSeed() const {
    return seed;
}
