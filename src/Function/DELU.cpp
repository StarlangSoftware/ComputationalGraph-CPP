//
// Created by Olcay YILDIZ on 26.01.2026.
//

#include "DELU.h"
#include "math.h"

using namespace std;

DELU::DELU(const double a, const double b, const double xc) {
    this->a = a;
    this->b = b;
    this->xc = xc;
    this->type = FunctionType::DELU;
}

DELU::DELU() {
    this->a = 1.0;
    this->b = 2.0;
    this->xc = 1.25643;
    this->type = FunctionType::DELU;
}

/**
 * Computes the DELU activation for the given value tensor.
 * @param value The tensor whose values are to be computed.
 * @return DELU(x).
 */
Tensor DELU::calculate(const Tensor &value) {
    vector<double> values;
    const vector<double> oldValues = value.getData();
    values.reserve(oldValues.size());
    for (const double &oldValue : oldValues) {
        if (oldValue < this->xc) {
            values.push_back(oldValue);
        } else {
            values.push_back((exp(this->a * oldValue) - 1) / this->b);
        }
    }
    return Tensor(values, value.getShape());
}

/**
 * Computes the derivative of the DELU activation function.
 * @param value output of the DELU(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor DELU::derivative(const Tensor &value, const Tensor &backward) {
    vector<double> values;
    const vector<double> oldValues = value.getData();
    const vector<double> backwardValues = backward.getData();
    values.reserve(oldValues.size());
    for (int i = 0; i < oldValues.size(); i++) {
        const auto oldValue = oldValues[i];
        auto backwardValue = backwardValues[i];
        if (oldValue > this->xc) {
            values.push_back(backwardValue);
        } else {
            values.push_back(backwardValue * ((oldValue * this->b + 1) * (this->a / this->b)));
        }
    }
    return Tensor(values, value.getShape());
}
