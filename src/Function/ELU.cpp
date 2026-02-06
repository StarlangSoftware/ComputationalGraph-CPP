//
// Created by Olcay YILDIZ on 26.01.2026.
//

#include "ELU.h"
#include "math.h"

ELU::ELU(const double a) {
    this->a = a;
    this->type = FunctionType::ELU;
}

ELU::ELU() {
    this->a = 1.0;
}

/**
 * Computes the ELU activation for the given tensor.
 * @param value The tensor whose values are to be computed.
 * @return ELU(x).
 */
Tensor ELU::calculate(const Tensor &value) {
    vector<double> values;
    const vector<double> oldValues = value.getData();
    values.reserve(oldValues.size());
    for (const double &oldValue : oldValues) {
        if (oldValue < 0.0) {
            values.push_back(a * (exp(oldValue) - 1));
        } else {
            values.push_back(oldValue);
        }
    }
    return Tensor(values, value.getShape());
}

/**
 * Computes the derivative of the ELU activation function.
 * @param value output of the ELU(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor ELU::derivative(const Tensor &value, const Tensor &backward) {
    vector<double> values;
    const vector<double> oldValues = value.getData();
    const vector<double> backwardValues = backward.getData();
    values.reserve(oldValues.size());
    for (int i = 0; i < oldValues.size(); i++) {
        const double oldValue = oldValues[i];
        double backwardValue = backwardValues[i];
        if (oldValue < 0.0) {
            values.push_back((oldValue + a) * backwardValue);
        } else {
            values.push_back(backwardValue);
        }
    }
    return Tensor(values, value.getShape());
}
