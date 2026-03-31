//
// Created by Olcay YILDIZ on 26.01.2026.
//

#include "Dropout.h"
#include "../ComputationalGraph.h"
#include "../Node/ComputationalNode.h"
#include <vector>

using namespace std;

Dropout::Dropout(const double p, const default_random_engine &randomEngine) {
  this->p = p;
  this->randomEngine = randomEngine;
  this->type = FunctionType::DROPOUT;
}

/**
 * Computes the dropout values for the given value tensor.
 * @param value The tensor whose values are to be computed.
 * @return Output tensor.
 */
Tensor Dropout::calculate(const Tensor &value) {
  this->mask.clear();
  const double multiplier = 1.0 / (1 - p);
  vector<double> values;
  const vector<double> oldValues = value.getData();
  uniform_real_distribution<> distribution(0, 1);
  values.reserve(oldValues.size());
  for (const auto &oldValue : oldValues) {
    double r = distribution(this->randomEngine);
    if (r > p) {
      this->mask.push_back(multiplier);
      values.push_back(oldValue * multiplier);
    } else {
      this->mask.push_back(0.0);
      values.push_back(0.0);
    }
  }
  return Tensor(values, value.getShape());
}

/**
 * Calculates the derivative of the dropout.
 * @param value output of the dropout function.
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor Dropout::derivative(const Tensor &value, const Tensor &backward) {
  return backward.hadamardProduct(Tensor(this->mask, value.getShape()));
}

ComputationalNode *
Dropout::addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                 bool isBiased, ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
