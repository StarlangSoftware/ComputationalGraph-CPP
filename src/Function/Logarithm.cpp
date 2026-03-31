//
// Created by Olcay YILDIZ on 26.01.2026.
//

#include "Logarithm.h"
#include "../ComputationalGraph.h"
#include "../Node/ComputationalNode.h"
#include <cmath>
#include <vector>

using namespace std;

Logarithm::Logarithm() { this->type = FunctionType::LOGARITHM; }

/**
 * Computes the element-wise logarithm of the given tensor.
 * @param value The tensor whose values are to be computed.
 * @return Log(x).
 */
Tensor Logarithm::calculate(const Tensor &value) {
  vector<double> values;
  const vector<double> oldValues = value.getData();
  values.reserve(oldValues.size());
  for (const double &oldValue : oldValues) {
    values.push_back(log(oldValue));
  }
  return Tensor(values, value.getShape());
}

/**
 * Computes the derivative of the Logarithm function.
 * @param value output of the Logarithm(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor Logarithm::derivative(const Tensor &value, const Tensor &backward) {
  vector<double> values;
  const vector<double> oldValues = value.getData();
  const vector<double> backwardValues = backward.getData();
  values.reserve(oldValues.size());
  for (int i = 0; i < oldValues.size(); i++) {
    values.push_back(backwardValues[i] / oldValues[i]);
  }
  return Tensor(values, value.getShape());
}

ComputationalNode *
Logarithm::addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                   bool isBiased, ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
