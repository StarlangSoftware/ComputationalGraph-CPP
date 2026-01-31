//
// Created by Olcay Taner YILDIZ on 30.01.2026.
//

#include "LinearPerceptronSingleUnit.h"
#include "../src/Function/Softmax.h"
#include "../src/Optimizer/StochasticGradientDescent.h"

LinearPerceptronSingleUnit::LinearPerceptronSingleUnit() {
}

vector<int> LinearPerceptronSingleUnit::getClassLabels(ComputationalNode *outputNode) {
    vector<int> labels;
    labels.push_back(0);
    return labels;
}

void LinearPerceptronSingleUnit::train(vector<Tensor> trainSet, NeuralNetworkParameter parameters) {
    Optimizer* optimizer = new StochasticGradientDescent(0.1, 0.99);
    auto input = new MultiplicationNode(false, true);
    inputNodes.push_back(input);
    const vector<double> initialWeights = {1.0, 1.0, 1.0, 1.0};
    const vector<int> weightsShape = {2, 2};
    const auto weightsTensor = Tensor(initialWeights, weightsShape);
    auto w = MultiplicationNode(weightsTensor);
    auto a = addEdge(input, &w, false);
    Softmax softmax;
    auto outputNode = addEdge(a, &softmax, false);
    Tensor dataTensor = trainSet[0];
    Tensor input1 = createInputTensor(dataTensor);
    input->setValue(input1);
    vector<int> calculatedClassLabels = forwardCalculation(false);
    vector<int> classes = {1};
    backpropagation(optimizer, classes);
    delete optimizer;
}

ClassificationPerformance LinearPerceptronSingleUnit::test(vector<Tensor> testSet) {
    return ClassificationPerformance(1.0);
}

