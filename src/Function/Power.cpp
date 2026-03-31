//
// Created by Olcay YILDIZ on 26.01.2026.
//

#include "Power.h"
#include "../ComputationalGraph.h"
#include "../Node/ComputationalNode.h"
#include <cmath>
#include <vector>

using namespace std;

Power::Power(double exponent) {
  this->exponent = exponent;
  this->type = FunctionType::POWER;
}

/**
 * Computes the element-wise power of the given tensor.
 * @param value The tensor whose values are to be computed.
 * @return x^exponent.
 */
Tensor Power::calculate(const Tensor &value) {
  vector<double> values;
  const vector<double> oldValues = value.getData();
  values.reserve(oldValues.size());
  for (const double &oldValue : oldValues) {
    values.push_back(pow(oldValue, exponent));
  }
  return Tensor(values, value.getShape());
}

/**
 * Computes the derivative of the Power function.
 * @param value output of the Power(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor Power::derivative(const Tensor &value, const Tensor &backward) {
  vector<double> values;
  const vector<double> oldValues = value.getData();
  const vector<double> backwardValues = backward.getData();
  values.reserve(oldValues.size());
  for (int i = 0; i < oldValues.size(); i++) {
    values.push_back(backwardValues[i] * exponent *
                     pow(oldValues[i], exponent - 1));
  }
  return Tensor(values, value.getShape());
}

ComputationalNode *
Power::addToGraph(const std::vector<ComputationalNode *> &inputNodes,
               bool isBiased, ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
