#include "Tanh.h"
#include "../ComputationalGraph.h"
#include "../Node/ComputationalNode.h"
#include <math.h>

/**
 * Computes the Tanh activation for the given tensor.
 * @param tensor The tensor whose values are to be computed.
 * @return Tanh(x).
 */
Tensor Tanh::calculate(const Tensor &tensor) {
  vector<double> values;
  vector<double> oldValues = tensor.getData();
  values.reserve(oldValues.size());
  for (double oldValue : oldValues) {
    values.push_back(tanh(oldValue));
  }
  return Tensor(values, tensor.getShape());
}

ComputationalNode *
Tanh::addToGraph(const std::vector<ComputationalNode *> &inputNodes, bool isBiased,
              ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}

/**
 * Computes the derivative of the Tanh activation function.
 * @param value output of the Tanh(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor Tanh::derivative(const Tensor &value, const Tensor &backward) {
  vector<double> values;
  vector<double> oldValues = value.getData();
  vector<double> backwardValues = backward.getData();
  values.reserve(oldValues.size());
  for (int i = 0; i < oldValues.size(); i++) {
    double oldValue = oldValues[i];
    double backwardValue = backwardValues[i];
    values.push_back((1 - oldValue * oldValue) * backwardValue);
  }
  return Tensor(values, value.getShape());
}
