//
// Created by Olcay YILDIZ on 27.01.2026.
//

#include "RandomInitialization.h"

/**
 * Random Uniform Initialization.
 * <p>
 * This method initializes the weights with small random values uniformly distributed
 * between -0.01 and 0.01. This is a basic initialization strategy used to break
 * symmetry between neurons.
 * </p>
 *
 * @param row    The number of rows in the matrix.
 * @param column The number of columns in the matrix.
 * @param randomEngine The {@link default_random_engine} instance used for generating values.
 * @return An {@link vector} containing the initialized weight values.
 */
vector<double> RandomInitialization::initialize(int row, int column, default_random_engine& randomEngine) {
    vector<double> data;
    uniform_real_distribution <> distribution (0, 1);
    data.reserve(row * column);
    for (int i = 0; i < row * column; i++) {
        data.push_back(-0.01 + 0.02 * distribution(randomEngine));
    }
    return data;
}
