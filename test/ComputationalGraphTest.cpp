//
// Created by Olcay Taner YILDIZ on 30.01.2026.
//

#include <Tensor.h>
#include <vector>

#include "../src/Optimizer/StochasticGradientDescent.h"
#include "LinearPerceptron.h"
#include "LinearPerceptronSingleUnit.h"

using namespace std;

void linearPerceptronSingleUnitTest() {
    vector<Tensor> trainSet;
    vector<double> data1 = {1.0, 1.0};
    vector<int> shape1 = {2};
    Tensor dataTensor = Tensor(data1, shape1);
    trainSet.push_back(dataTensor);
    LinearPerceptronSingleUnit graph;
    graph.train(trainSet, NeuralNetworkParameter(1, 1, nullptr));
}

void linearPerceptronTest() {
    vector<Tensor> trainSet;
    vector<Tensor> testSet;
    LinearPerceptron graph;
    graph.createIrisDataset(trainSet, testSet);
    graph.train(trainSet, NeuralNetworkParameter(1, 10, new StochasticGradientDescent(0.1, 0.99)));
    ClassificationPerformance performance = graph.test(testSet);
}

int main() {
    //linearPerceptronSingleUnitTest();
    linearPerceptronTest();
}