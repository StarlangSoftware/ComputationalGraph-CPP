//
// Created by Olcay YILDIZ on 28.01.2026.
//

#include "Adam.h"

Adam::Adam(const double learningRate, const double etaDecrease, const double beta1, const double beta2, const double epsilon) : SGDMomentum(learningRate, etaDecrease, beta1){
    this->beta2 = beta2;
    this->epsilon = epsilon;
    this->currentBeta1 = 1;
    this->currentBeta2 = 1;
}

/**
 * Calculates the gradient updates using the Adam optimization algorithm.
 * This implementation follows a multi-pass approach:
 * <ol>
 * <li><b>First Pass:</b> Calculates the weighted current gradients for both the first moment (momentum)
 * and the second moment (velocity/squared gradients).</li>
 * <li><b>Second Pass (Conditional):</b> If historical data exists, adds the decayed previous
 * momentum and velocity values to the current ones.</li>
 * <li><b>State Update:</b> Stores the raw calculated moments into the history maps.</li>
 * <li><b>Bias Correction:</b> Normalizes the moments by dividing them by <code>(1 - (beta)^t)</code>
 * to account for initialization bias.</li>
 * <li><b>Final Pass:</b> Computes the parameter update using the adaptive learning rate formula:
 * <code>(new_momentum / (sqrt(new_velocity) + epsilon)) * learningRate</code>.</li>
 * </ol>
 *
 * @param node The node whose gradients are to be set.
 */
vector<double> Adam::calculate(const ComputationalNode& node) {
    int backwardSize = node.getBackward().getData().size();
    vector<double> newValuesMomentum;
    newValuesMomentum.reserve(backwardSize);
    vector<double> newValuesVelocity;
    newValuesVelocity.reserve(backwardSize);
    for (int i = 0; i < backwardSize; i++) {
        const double backwardValue = node.getBackward().getData()[i];
        newValuesMomentum.push_back((1 - momentum) * backwardValue);
        newValuesVelocity.push_back((1 - beta2) * (backwardValue * backwardValue));
    }
    if (momentumMap.contains(node)) {
        for (int i = 0; i < newValuesVelocity.size(); i++) {
            newValuesVelocity[i] += beta2 * velocityMap[node][i];
            newValuesMomentum[i] += momentum * momentumMap[node][i];
        }
    }
    vector<double> momentumValues;
    vector<double> velocityValues;
    for (int i = 0; i < backwardSize; i++) {
        momentumValues.push_back(newValuesMomentum[i]);
        velocityValues.push_back(newValuesVelocity[i]);
    }
    momentumMap[node] = momentumValues;
    velocityMap[node] = velocityValues;
    for (int i = 0; i < newValuesVelocity.size(); i++) {
        newValuesMomentum[i] /= 1 - currentBeta1;
        newValuesVelocity[i] /= 1 - currentBeta2;
    }
    vector<double> newValues;
    newValues.reserve(newValuesMomentum.size());
    for (int i = 0; i < newValuesMomentum.size(); i++) {
        newValues.push_back((newValuesMomentum[i] / (sqrt(newValuesVelocity[i]) + epsilon)) * learningRate);
    }
    return newValues;
}

/**
 * Sets the gradients for the given node using the Adam optimization algorithm.
 * @param node The node whose gradients are to be set.
 */
void Adam::setGradients(ComputationalNode &node) {
    node.setBackward(Tensor(calculate(node), node.getBackward().getShape()));
}

/**
 * Updates the values of all learnable nodes in the graph.
 * @param nodeMap A map of nodes to their children.
 */
void Adam::updateValues(map<ComputationalNode*, vector<ComputationalNode*>> &nodeMap) {
    currentBeta1 *= momentum;
    currentBeta2 *= beta2;
    Optimizer::updateValues(nodeMap);
}
