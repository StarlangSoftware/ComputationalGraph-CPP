#include "Tanh.hpp"
#include <cmath>

Matrix Tanh::calculate(const Matrix& matrix) const {
    Matrix result(matrix.getRow(), matrix.getColumn());

    for (int i = 0; i < matrix.getRow(); ++i) {
        for (int j = 0; j < matrix.getColumn(); ++j) {
            double val = matrix.getValue(i, j);
            result.setValue(i, j, std::tanh(val));
        }
    }

    return result;
}

Matrix Tanh::derivative(const Matrix& matrix) const {
    Matrix result(matrix.getRow(), matrix.getColumn());

    for (int i = 0; i < matrix.getRow(); ++i) {
        for (int j = 0; j < matrix.getColumn(); ++j) {
            double val = matrix.getValue(i, j);
            double tanh_val = std::tanh(val);
            result.setValue(i, j, 1.0 - tanh_val * tanh_val);
        }
    }

    return result;
}
