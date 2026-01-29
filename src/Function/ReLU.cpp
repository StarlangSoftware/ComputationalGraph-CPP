#include "ReLU.h"

#include <Matrix.h>

/**
 * Computes the ReLU activation for the given tensor.
 * @param value The tensor whose values are to be computed.
 * @return ReLU(x).
 */
Tensor calculate(const Tensor& value) {
    vector<double> values;
    const vector<double> oldValues = value.getData();
    values.reserve(oldValues.size());
    for (const double& oldValue : oldValues) {
        values.push_back(max(oldValue, 0.0));
    }
    return Tensor(values, value.getShape());
}

/**
 * Computes the derivative of the ReLU activation function.
 * @param tensor output of the ReLU(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor derivative(const Tensor& tensor, const Tensor& backward) {
    vector<double> values;
    const vector<double> oldValues = tensor.getData();
    const vector<double> backwardValues = backward.getData();
    values.reserve(oldValues.size());
    for (int i = 0; i < oldValues.size(); i++) {
        const double oldValue = oldValues[i];
        const double backwardValue = backwardValues[i];
        if (oldValue > 0) {
            values.push_back(backwardValue);
        } else {
            values.push_back(0.0);
        }
    }
    return Tensor(values, tensor.getShape());
}
