#include "Softmax.h"
#include <math.h>
#include <Matrix.h>

/**
 * Computes the Softmax activation for the given tensor.
 * @param tensor The tensor whose values are to be computed.
 * @return Softmax(x).
 */
Tensor calculate(const Tensor &tensor) {
    vector<double> values;
    vector<double> oldValues = tensor.getData();
    int lastDimensionSize = tensor.getShape()[tensor.getShape().size() - 1];
    double sum = 0.0;
    vector<double> sumList;
    for (int i = 0; i < oldValues.size(); i++) {
        sum += exp(oldValues[i]);
        if ((i + 1) % lastDimensionSize == 0) {
            sumList.push_back(sum);
            sum = 0.0;
        }
    }
    values.reserve(oldValues.size());
    for (int i = 0; i < oldValues.size(); i++) {
        values.push_back(exp(sumList[i]) / sumList[i / lastDimensionSize]);
    }
    return Tensor(values, tensor.getShape());
}

/**
 * Computes the derivative of the Softmax activation function.
 * @param tensor output of the Softmax(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor derivative(const Tensor &tensor, const Tensor &backward) {
    int lastDimensionSize = tensor.getShape()[tensor.getShape().size() - 1];
    vector<double> values;
    vector<double> oldValuesTensor = tensor.getData();
    vector<double> oldValuesBackward = backward.getData();
    double total = 0.0;
    for (int i = 0; i < oldValuesTensor.size(); i++) {
        total += oldValuesTensor[i] * oldValuesBackward[i];
        if ((i + 1) % lastDimensionSize == 0) {
            int startIndex = i / lastDimensionSize;
            for (int j = 0; j < lastDimensionSize; j++) {
                values.push_back(oldValuesBackward[startIndex * lastDimensionSize + j] - total);
            }
            total = 0.0;
        }
    }
    return tensor.hadamardProduct(Tensor(values, tensor.getShape()));
}
