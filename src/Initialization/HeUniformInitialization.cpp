//
// Created by Olcay YILDIZ on 27.01.2026.
//
#include "HeUniformInitialization.h"
#include "math.h"

/**
 * He Uniform Initialization.
 * <p>
 * This method initializes weights using a uniform distribution, which is typically
 * optimized for layers with ReLU activation functions. It helps in maintaining
 * the variance of activations throughout the network layers.
 * </p>
 *
 * @param row    The number of rows in the matrix (typically represents the output size / number of neurons).
 * @param column The number of columns in the matrix (typically represents the input size / fan-in).
 * @param randomEngine The {@link default_random_engine} instance used for generating values (allows for reproducibility).
 * @return An {@link vector} of Doubles containing the initialized weight values.
 */
vector<double> HeUniformInitialization::initialize(int row, int column, default_random_engine& randomEngine) {
    vector<double> data;
    uniform_real_distribution <> distribution (0, 1);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            data.push_back((sqrt(6.0 / column) + sqrt(6.0 / row)) * distribution(randomEngine));
        }
    }
    return data;
}
