//
// Created by Olcay YILDIZ on 1.02.2026.
//

#include "DeepNetwork.h"
#include "../src/Function/Sigmoid.h"
#include "../src/Function/Softmax.h"
#include "../src/Initialization/RandomInitialization.h"
#include "../src/Optimizer/StochasticGradientDescent.h"

DeepNetwork::DeepNetwork() = default;

void DeepNetwork::train(vector<Tensor> &trainSet,
                        NeuralNetworkParameter &parameters) {
  Optimizer *optimizer = parameters.getOptimizer();
  auto input = new MultiplicationNode(false, true);
  inputNodes.push_back(input);
  int number_of_input_units_with_biased = 5;
  int numberOfHiddenUnitsInLayer1 = 6;
  auto randomEngine = default_random_engine(parameters.getSeed());
  Initialization *initialization = parameters.getInitialization();
  vector<double> initialWeights =
      initialization->initialize(number_of_input_units_with_biased,
                                 numberOfHiddenUnitsInLayer1, randomEngine);
  const vector<int> weightsShape = {number_of_input_units_with_biased,
                                    numberOfHiddenUnitsInLayer1};
  const auto weightsTensor = Tensor(initialWeights, weightsShape);
  auto w = new MultiplicationNode(weightsTensor);
  auto a = addMultiplicationEdge(input, w, false);
  auto sigmoid = new Sigmoid();
  auto aSigmoid = addFunctionEdge(a, sigmoid, true);
  int numberOfHiddenUnitsInLayer2 = 10;
  vector<double> initialWeights2 =
      initialization->initialize(numberOfHiddenUnitsInLayer1 + 1,
                                 numberOfHiddenUnitsInLayer2, randomEngine);
  const vector<int> weightsShape2 = {numberOfHiddenUnitsInLayer1 + 1,
                                     numberOfHiddenUnitsInLayer2};
  const auto weightsTensor2 = Tensor(initialWeights2, weightsShape2);
  auto w2 = new MultiplicationNode(weightsTensor2);
  auto a2 = addMultiplicationEdge(aSigmoid, w2, false);
  auto sigmoid2 = new Sigmoid();
  auto aSigmoid2 = addFunctionEdge(a2, sigmoid2, true);
  int number_of_classes = 3;
  vector<double> initialWeights3 = initialization->initialize(
      numberOfHiddenUnitsInLayer2 + 1, number_of_classes, randomEngine);
  const vector<int> weightsShape3 = {numberOfHiddenUnitsInLayer2 + 1,
                                     number_of_classes};
  const auto weightsTensor3 = Tensor(initialWeights3, weightsShape3);
  auto w3 = new MultiplicationNode(weightsTensor3);
  auto a3 = addMultiplicationEdge(aSigmoid2, w3, false);
  auto softmax = new Softmax();
  outputNode = addFunctionEdge(a3, softmax, false);
  for (int i = 0; i < parameters.getEpoch(); i++) {
    for (const auto &instance : trainSet) {
      input->setValue(createInputTensor(instance));
      vector<int> calculatedClasses = forwardCalculation();
      vector<int> index = {instance.getShape()[0] - 1};
      vector<int> classes = {(int)instance.getValue(index)};
      backpropagation(optimizer, classes);
    }
  }
}
