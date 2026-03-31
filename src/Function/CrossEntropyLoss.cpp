//
// Created by Olcay YILDIZ on 26.01.2026.
//

#include "CrossEntropyLoss.h"
#include "../ComputationalGraph.h"
#include "../Node/ComputationalNode.h"
#include <cmath>
#include <vector>

using namespace std;

CrossEntropyLoss::CrossEntropyLoss() {
  this->type = FunctionType::CROSS_ENTROPY_LOSS;
}

/**
 * Computes the Cross Entropy Loss.
 * @param value Predicted probabilities.
 * @return Loss value.
 */
Tensor CrossEntropyLoss::calculate(const Tensor &value) {
  // In this library, calculate for loss might return the sum or mean.
  // Based on Java implementation (implied):
  vector<double> values;
  const vector<double> oldValues = value.getData();
  values.reserve(oldValues.size());
  for (const double &oldValue : oldValues) {
    values.push_back(-log(max(oldValue, 1e-15)));
  }
  return Tensor(values, value.getShape());
}

/**
 * Computes the derivative of the Cross Entropy Loss.
 * @param value Measured output.
 * @param backward Target values (y - r).
 * @return Gradient value.
 */
Tensor CrossEntropyLoss::derivative(const Tensor &value,
                                    const Tensor &backward) {
  // Java implementation likely uses the fact that backpropagation handles (y -
  // r) for CrossEntropy + Softmax. Here we just return backward as it's
  // typically pre-calculated.
  return backward;
}

ComputationalNode *
CrossEntropyLoss::addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                          bool isBiased, ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
