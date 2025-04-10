#include "Softmax.h"
#include <cmath>
#include <Matrix.h>

Matrix Softmax::calculate(const Matrix& matrix) const {
    Matrix result(matrix.getRow(), matrix.getColumn());

    for (int i = 0; i < matrix.getRow(); ++i) {
        double sum_exp = 0.0;
        for (int k = 0; k < matrix.getColumn(); ++k) {
            sum_exp += std::exp(matrix.getValue(i, k));
        }
        for (int k = 0; k < matrix.getColumn(); ++k) {
            double val = std::exp(matrix.getValue(i, k)) / sum_exp;
            result.setValue(i, k, val);
        }
    }

    return result;
}

Matrix Softmax::derivative(const Matrix& matrix) const {
    // Derivative implementation depends on usage (often involves Jacobian matrix).
    // Placeholder: Not implemented as per original Java code.
    return Matrix(matrix.getRow(), matrix.getColumn());
}
