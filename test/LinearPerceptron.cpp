//
// Created by Olcay YILDIZ on 31.01.2026.
//

#include "LinearPerceptron.h"
#include "../src/Optimizer/StochasticGradientDescent.h"
#include "../src/Function/Softmax.h"
#include "../src/Initialization/RandomInitialization.h"

using namespace std;

LinearPerceptron::LinearPerceptron() {
}

void LinearPerceptron::train(vector<Tensor> trainSet, NeuralNetworkParameter parameters) {
    Optimizer* optimizer = parameters.getOptimizer();
    auto input = new MultiplicationNode(false, true);
    inputNodes.push_back(input);
    int number_of_input_units_with_biased = 5;
    int number_of_classes = 3;
    auto randomEngine = default_random_engine(parameters.getSeed());
    Initialization* initialization = parameters.getInitialization();
    vector<double> initialWeights = initialization->initialize(number_of_input_units_with_biased, number_of_classes, randomEngine);
    const vector<int> weightsShape = {number_of_input_units_with_biased, number_of_classes};
    const auto weightsTensor = Tensor(initialWeights, weightsShape);
    auto w = new MultiplicationNode(weightsTensor);
    auto a = addEdge(input, w, false);
    auto softmax = new Softmax();
    auto outputNode = addEdge(a, softmax, false);
    for (int i = 0; i < parameters.getEpoch(); i++) {
        for (const auto& instance : trainSet) {
            input->setValue(createInputTensor(instance));
            vector<int> calculatedClasses = forwardCalculation();
            vector<int> index = {instance.getShape()[0] - 1};
            vector<int> classes = {(int)instance.getValue(index)};
            backpropagation(optimizer, classes);
        }
    }
    delete optimizer;}

ClassificationPerformance LinearPerceptron::test(vector<Tensor> testSet) {
    int count = 0, total = 0;
    for (const auto & instance : testSet) {
        ComputationalNode* input = inputNodes[0];
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

vector<int> LinearPerceptron::getClassLabels(ComputationalNode *outputNode) {
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
