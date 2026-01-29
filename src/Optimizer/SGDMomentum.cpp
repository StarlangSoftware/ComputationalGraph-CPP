//
// Created by Olcay YILDIZ on 28.01.2026.
//

#include "SGDMomentum.h"

SGDMomentum::SGDMomentum(const double learningRate, const double etaDecrease, const double momentum) : Optimizer(learningRate, etaDecrease){
    this->momentum = momentum;
}

/**
 * Calculates the new gradients by combining the current gradient with the previous velocity.
 * It updates the internal velocity state and modifies the node's backward tensor
 * to reflect the momentum-adjusted update step.
 *
 * @param node The node whose gradients are to be set.
 */
void SGDMomentum::setGradients(ComputationalNode &node) {
    int backwardSize = node.getBackward().getData().size();
    vector<double> newValues;
    newValues.reserve(backwardSize);
    for (int i = 0; i < backwardSize; i++) {
        newValues.push_back((1 - momentum) * node.getBackward().getData()[i]);
    }
    if (velocityMap.contains(node)) {
        for (int i = 0; i < newValues.size(); i++) {
            newValues[i] += velocityMap[node][i] * momentum;
        }
    }
    vector<double> velocity;
    velocity.reserve(backwardSize);
    for (int i = 0; i < backwardSize; i++) {
        velocity.push_back(newValues[i]);
    }
    velocityMap[node] = velocity;
    for (double & newValue : newValues) {
        newValue *= learningRate;
    }
    node.setBackward(Tensor(newValues, node.getBackward().getShape()));
}
