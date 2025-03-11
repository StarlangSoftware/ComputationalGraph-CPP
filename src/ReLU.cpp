#include "ReLU.hpp"

Matrix ReLU::calculate(const Matrix& matrix) const {
    Matrix result(matrix.getRow(), matrix.getColumn());

    for (int i = 0; i < matrix.getRow(); ++i) {
        for (int j = 0; j < matrix.getColumn(); ++j) {
            double val = matrix.getValue(i, j);
            result.setValue(i, j, val > 0 ? val : 0.0);
        }
    }

    return result;
}

Matrix ReLU::derivative(const Matrix& matrix) const {
    Matrix result(matrix.getRow(), matrix.getColumn());

    for (int i = 0; i < matrix.getRow(); ++i) {
        for (int j = 0; j < matrix.getColumn(); ++j) {
            double val = matrix.getValue(i, j);
            result.setValue(i, j, val > 0.0 ? 1.0 : 0.0);
        }
    }

    return result;
}
