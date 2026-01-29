//
// Created by Olcay YILDIZ on 28.01.2026.
//

#include "StochasticGradientDescent.h"

StochasticGradientDescent::StochasticGradientDescent(const double learningRate, const double etaDecrease) : Optimizer(learningRate, etaDecrease){
}

/**
 * Sets the gradients (backward values) of the node to the learning rate times the backward values.
 * @param node The node whose gradients are to be set.
 */
void StochasticGradientDescent::setGradients(ComputationalNode &node) {
    vector<double> values;
    vector<double> backward = node.getBackward().getData();
    values.reserve(backward.size());
    for (const double& value : backward) {
        values.push_back(value * learningRate);
    }
    node.setBackward(Tensor(values, node.getBackward().getShape()));
}
