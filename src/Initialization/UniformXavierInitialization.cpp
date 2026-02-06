//
// Created by Olcay YILDIZ on 27.01.2026.
//

#include "UniformXavierInitialization.h"
#include "math.h"

/**
 * Xavier Uniform Initialization.
 * <p>
 * This method initializes weights using a uniform distribution within the range
 * [-limit, limit], where the limit is sqrt(6 / (fan_in + fan_out)).
 * This strategy is designed to keep the scale of the gradients roughly the same
 * in all layers and is commonly used with Sigmoid or Tanh activation functions.
 * </p>
 *
 * @param row    The number of rows in the matrix (typically represents fan-out / output size).
 * @param column The number of columns in the matrix (typically represents fan-in / input size).
 * @param randomEngine The {@link default_random_engine} instance used for generating values.
 * @return An {@link vector} containing the initialized weight values.
 */
vector<double> UniformXavierInitialization::initialize(int row, int column, default_random_engine& randomEngine) {
    vector<double> data;
    uniform_real_distribution <> distribution (0, 1);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            data.push_back((2 * distribution(randomEngine) - 1) * sqrt(6.0 / (row + column)));
        }
    }
    return data;
}
