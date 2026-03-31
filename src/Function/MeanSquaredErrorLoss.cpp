//
// Created by Olcay YILDIZ on 26.01.2026.
//

#include "MeanSquaredErrorLoss.h"
#include "../ComputationalGraph.h"
#include "../Node/ComputationalNode.h"
#include <vector>

using namespace std;

MeanSquaredErrorLoss::MeanSquaredErrorLoss() {
  this->type = FunctionType::MSE_LOSS;
}

/**
 * Computes the MSE Loss.
 * @param value Predicted values.
 * @return Squared errors.
 */
Tensor MeanSquaredErrorLoss::calculate(const Tensor &value) {
  vector<double> values;
  const vector<double> oldValues = value.getData();
  values.reserve(oldValues.size());
  for (const double &oldValue : oldValues) {
    values.push_back(0.5 * oldValue * oldValue);
  }
  return Tensor(values, value.getShape());
}

/**
 * Computes the derivative of the MSE Loss.
 * @param value Measured output.
 * @param backward Target values (y - r).
 * @return Gradient value.
 */
Tensor MeanSquaredErrorLoss::derivative(const Tensor &value,
                                        const Tensor &backward) {
  return backward;
}

ComputationalNode *MeanSquaredErrorLoss::addToGraph(
    const std::vector<ComputationalNode *> &inputNodes, bool isBiased,
    ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
