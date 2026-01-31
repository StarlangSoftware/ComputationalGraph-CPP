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

void NeuralNetwork::createIrisDataset(vector<Tensor> &trainSet, vector<Tensor> &testSet) {
    vector<int> strides = {5};
    for (int i = 0; i < 150; i++) {
        vector<double> inputData;
        inputData.reserve(5);
        for (int j = 0; j < 5; j++) {
            inputData.push_back(irisData[i][j]);
        }
        Tensor input = Tensor(inputData, strides);
        if (i % 5 != 0) {
            trainSet.emplace_back(input);
        } else {
            testSet.emplace_back(input);
        }
    }
}
