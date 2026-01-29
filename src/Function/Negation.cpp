//
// Created by Olcay YILDIZ on 26.01.2026.
//

#include "Negation.h"

/**
 * Negates the values of the given tensor.
 * @param value The tensor whose values are to be negated.
 * @return The negated tensor.
 */
Tensor Negation::calculate(const Tensor &value) {
    vector<double> values;
    vector<double> oldValues = value.getData();
    values.reserve(oldValues.size());
    for (const auto &oldValue : oldValues) {
        values.push_back(-oldValue);
    }
    return Tensor(values, value.getShape());
}

/**
 * Calculates the derivative of the Negation function.
 * @param value output of the Negation function.
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor Negation::derivative(const Tensor &value, const Tensor &backward) {
    vector<double> values;
    vector<double> backwardValues = backward.getData();
    values.reserve(backwardValues.size());
    for (const auto &backward_value : backwardValues) {
        values.push_back(-backward_value);
    }
    return Tensor(values, value.getShape());
}
