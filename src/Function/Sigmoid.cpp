#include "Sigmoid.h"
#include "../ComputationalGraph.h"
#include "../Node/ComputationalNode.h"
#include <cmath>

/**
 * Computes the Sigmoid activation for the given tensor.
 * @param value The tensor whose values are to be computed.
 * @return Sigmoid(x).
 */
Tensor Sigmoid::calculate(const Tensor &value) {
  vector<double> values;
  const vector<double> tensorValues = value.getData();
  values.reserve(tensorValues.size());
  for (const double &tensorValue : tensorValues) {
    double sigmoid = 1.0 / (1.0 + exp(-tensorValue));
    values.push_back(sigmoid);
  }
  return Tensor(values, value.getShape());
}

/**
 * Computes the derivative of the Sigmoid activation function.
 * @param tensor output of the Sigmoid(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor Sigmoid::derivative(const Tensor &tensor, const Tensor &backward) {
  vector<double> values;
  const vector<double> tensorValues = tensor.getData();
  const vector<double> backwardValues = backward.getData();
  values.reserve(tensorValues.size());
  for (int i = 0; i < tensorValues.size(); i++) {
    const double val = tensorValues[i];
    const double derivative = val * (1 - val);
    const double backwardValue = backwardValues[i];
    values.push_back(derivative * backwardValue);
  }
  return Tensor(values, tensor.getShape());
}

ComputationalNode *
Sigmoid::addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                 bool isBiased, ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
