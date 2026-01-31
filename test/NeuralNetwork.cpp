//
// Created by Olcay Taner YILDIZ on 30.01.2026.
//

#include "NeuralNetwork.h"

Tensor NeuralNetwork::createInputTensor(const Tensor& instance) {
    vector<double> data;
    for (int i = 0; i < instance.getShape()[0] - 1; i++) {
        data.emplace_back(instance.getValue({i}));
    }
    return Tensor(data, {1, instance.getShape()[0] - 1});
}