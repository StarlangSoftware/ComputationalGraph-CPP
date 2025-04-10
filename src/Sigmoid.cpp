#include "Sigmoid.h"

#include <Matrix.h>

Matrix Sigmoid::calculate(const Matrix& matrix) const {
    Matrix result(matrix.getRow(), matrix.getColumn());

    for (int i = 0; i < matrix.getRow(); ++i) {
        for (int j = 0; j < matrix.getColumn(); ++j) {
            double val = matrix.getValue(i, j);
            result.setValue(i, j, 1.0 / (1.0 + std::exp(-val)));
        }
    }

    return result;
}

Matrix Sigmoid::derivative(const Matrix& matrix) const {
    Matrix result(matrix.getRow(), matrix.getColumn());

    for (int i = 0; i < matrix.getRow(); ++i) {
        for (int j = 0; j < matrix.getColumn(); ++j) {
            double val = matrix.getValue(i, j);
            result.setValue(i, j, val * (1.0 - val));
        }
    }

    return result;
}
