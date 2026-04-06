//
// Created by Olcay Taner YILDIZ on 30.01.2026.
//

#include "NeuralNetwork.h"

Tensor NeuralNetwork::createInputTensor(const Tensor &instance) {
  vector<double> data;
  for (int i = 0; i < instance.getShape()[0] - 1; i++) {
    data.emplace_back(instance.getValue({i}));
  }
  return Tensor(data, {1, instance.getShape()[0] - 1});
}

void NeuralNetwork::createIrisDataset(vector<Tensor> &trainSet,
                                      vector<Tensor> &testSet) {
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

ClassificationPerformance NeuralNetwork::test(const vector<Tensor> &testSet) {
  int count = 0, total = 0;
  for (const auto &instance : testSet) {
    ComputationalNode *input = inputNodes[0];
    input->setValue(createInputTensor(instance));
    vector<int> output = predict();
    int classLabel = output[0];
    vector<int> index = {instance.getShape()[0] - 1};
    if (classLabel == instance.getValue(index)) {
      count++;
    }
    total++;
  }
  return ClassificationPerformance(count / (total + 0.0));
}

vector<int> NeuralNetwork::getClassLabels(ComputationalNode *outputNode) {
  vector<int> classIndices;
  Tensor outputValue = outputNode->getValue();
  int cols = outputValue.getShape()[1];
  double maxValue = -1;
  int labelIndex = -1;
  vector<int> indices = {0, 0};
  for (int i = 0; i < cols; i++) {
    indices[1] = i;
    double value = outputValue.getValue(indices);
    if (value > maxValue) {
      maxValue = value;
      labelIndex = i;
    }
  }
  classIndices.push_back(labelIndex);
  return classIndices;
}
