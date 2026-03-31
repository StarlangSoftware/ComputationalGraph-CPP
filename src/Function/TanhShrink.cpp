//
// Created by Olcay YILDIZ on 26.01.2026.
//

#include "TanhShrink.h"
#include "../ComputationalGraph.h"
#include "../Node/ComputationalNode.h"
#include "Negation.h"
#include "Tanh.h"
#include <vector>

using namespace std;

TanhShrink::TanhShrink() { this->type = FunctionType::TANH_SHRINK; }

/**
 * Computes the TanhShrink activation (x - Tanh(x)) for the given tensor.
 * @param value The tensor whose values are to be computed.
 * @return TanhShrink(x).
 */
Tensor TanhShrink::calculate(const Tensor &value) {
  Tanh tanh;
  Tensor tanhValue = tanh.calculate(value);
  vector<double> values;
  vector<double> xValues = value.getData();
  vector<double> tValues = tanhValue.getData();
  values.reserve(xValues.size());
  for (int i = 0; i < xValues.size(); i++) {
    values.push_back(xValues[i] - tValues[i]);
  }
  return Tensor(values, value.getShape());
}

/**
 * Computes the derivative of the TanhShrink function.
 * @param value output of the TanhShrink(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor TanhShrink::derivative(const Tensor &value, const Tensor &backward) {
  return backward; // Placeholder, subgraph is primary.
}

ComputationalNode *
TanhShrink::addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                    bool isBiased, ComputationalGraph *graph) {
  Tanh *tanh = new Tanh();
  ComputationalNode *node2 = tanh->addToGraph(inputNodes, isBiased, graph);
  Negation *negation = new Negation();
  ComputationalNode *negatedNode = negation->addToGraph({node2}, false, graph);
  return graph->addAdditionEdge(inputNodes[0], negatedNode, isBiased);
}
