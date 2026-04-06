//
// Created by Olcay Taner YILDIZ on 30.01.2026.
//

#include <Tensor.h>
#include <vector>

#include "../src/Optimizer/StochasticGradientDescent.h"
#include "DeepNetwork.h"
#include "LinearPerceptron.h"
#include "LinearPerceptronSingleUnit.h"
#include "MultiLayerPerceptron.h"

#include "../src/Function/Logarithm.h"
#include "../src/Function/Power.h"
#include "../src/Function/SiLU.h"
#include "../src/Function/TanhShrink.h"
#include <cassert>
#include <iostream>

using namespace std;

void testLogarithm() {
  Logarithm logFunc;
  vector<double> data = {exp(1.0), exp(2.0)};
  Tensor t(data, {2});
  Tensor result = logFunc.calculate(t);
  assert(abs(result.getData()[0] - 1.0) < 1e-5);
  assert(abs(result.getData()[1] - 2.0) < 1e-5);
  cout << "Logarithm Test Passed!" << endl;
}

void testPower() {
  Power powerFunc(2.0);
  vector<double> data = {2.0, 3.0};
  Tensor t(data, {2});
  Tensor result = powerFunc.calculate(t);
  assert(abs(result.getData()[0] - 4.0) < 1e-5);
  assert(abs(result.getData()[1] - 9.0) < 1e-5);
  cout << "Power Test Passed!" << endl;
}

void testSiLU() {
  SiLU silu;
  vector<double> data = {0.0};
  Tensor t(data, {1});
  Tensor result = silu.calculate(t);
  assert(abs(result.getData()[0] - 0.0) < 1e-5);
  cout << "SiLU Test Passed!" << endl;
}

void testTanhShrink() {
  TanhShrink ts;
  vector<double> data = {0.0};
  Tensor t(data, {1});
  Tensor result = ts.calculate(t);
  assert(abs(result.getData()[0] - 0.0) < 1e-5);
  cout << "TanhShrink Test Passed!" << endl;
}

void linearPerceptronSingleUnitTest() {
  vector<Tensor> trainSet;
  vector<double> data1 = {1.0, 1.0};
  vector<int> shape1 = {2};
  Tensor dataTensor = Tensor(data1, shape1);
  trainSet.push_back(dataTensor);
  LinearPerceptronSingleUnit graph;
  NeuralNetworkParameter params(1, 1, nullptr);
  graph.train(trainSet, params);
}

void linearPerceptronTest() {
  vector<Tensor> trainSet;
  vector<Tensor> testSet;
  LinearPerceptron graph;
  graph.createIrisDataset(trainSet, testSet);
  NeuralNetworkParameter params(1, 10,
                                new StochasticGradientDescent(0.1, 0.99));
  graph.train(trainSet, params);
  ClassificationPerformance performance = graph.test(testSet);
  cout << "Linear Perceptron Test Finished!" << endl;
}

void multiLayerPerceptronTest() {
  vector<Tensor> trainSet;
  vector<Tensor> testSet;
  MultiLayerPerceptron graph;
  graph.createIrisDataset(trainSet, testSet);
  NeuralNetworkParameter params(1, 10,
                                new StochasticGradientDescent(0.1, 0.99));
  graph.train(trainSet, params);
  ClassificationPerformance performance = graph.test(testSet);
}

void deepNetworkTest() {
  vector<Tensor> trainSet;
  vector<Tensor> testSet;
  DeepNetwork graph;
  graph.createIrisDataset(trainSet, testSet);
  NeuralNetworkParameter params(1, 10,
                                new StochasticGradientDescent(0.1, 0.99));
  graph.train(trainSet, params);
  ClassificationPerformance performance = graph.test(testSet);
}

int main() {
  testLogarithm();
  testPower();
  testSiLU();
  testTanhShrink();
  linearPerceptronSingleUnitTest();
  linearPerceptronTest();
  multiLayerPerceptronTest();
  deepNetworkTest();
  return 0;
}