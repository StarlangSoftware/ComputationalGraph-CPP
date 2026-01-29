//
// Created by Olcay YILDIZ on 28.01.2026.
//

#include "AdamW.h"

AdamW::AdamW(const double learningRate, const double etaDecrease, const double beta1, const double beta2, const double epsilon,
    double weightDecay) : Adam(learningRate, etaDecrease, beta1, beta2, epsilon) {
    this->weightDecay = weightDecay;
}

/**
 * Sets the gradients for the given node using the AdamW optimization algorithm.
 * @param node The node whose gradients are to be set.
 */
void AdamW::setGradients(ComputationalNode &node) {
    vector<double> gradients = calculate(node);
    vector<double> values = node.getValue().getData();
    for (int i = 0; i < gradients.size(); i++) {
        gradients[i] += learningRate * weightDecay * values[i];
    }
    node.setBackward(Tensor(gradients, node.getBackward().getShape()));
}
