//
// Created by Olcay Taner YILDIZ on 30.01.2026.
//

#include <Tensor.h>
#include <vector>

#include "LinearPerceptronSingleUnit.h"

void linearPerceptronSingleUnitTest() {
    std::vector<Tensor> trainSet;
    vector<double> data1 = {1.0, 1.0};
    vector<int> shape1 = {2};
    Tensor dataTensor = Tensor(data1, shape1);
    trainSet.push_back(dataTensor);
    LinearPerceptronSingleUnit graph;
    graph.train(trainSet, NeuralNetworkParameter(1, 1, nullptr));
}

int main() {
    linearPerceptronSingleUnitTest();
}