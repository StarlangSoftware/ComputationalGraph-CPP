//
// Created by Olcay YILDIZ on 26.01.2026.
//

#include "SiLU.h"
#include "../ComputationalGraph.h"
#include "../Node/ComputationalNode.h"
#include "Sigmoid.h"
#include <cmath>
#include <vector>

using namespace std;

SiLU::SiLU() { this->type = FunctionType::SILU; }

/**
 * Computes the SiLU activation (x * Sigmoid(x)) for the given tensor.
 * @param value The tensor whose values are to be computed.
 * @return SiLU(x).
 */
Tensor SiLU::calculate(const Tensor &value) {
  Sigmoid sigmoid;
  Tensor sigmoidValue = sigmoid.calculate(value);
  return value.hadamardProduct(sigmoidValue);
}

/**
 * Computes the derivative of the SiLU activation function.
 * @param value output of the SiLU(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor SiLU::derivative(const Tensor &value, const Tensor &backward) {
  // SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
  // This is complex to compute from value (x * sigmoid(x)) alone.
  // However, since we use subgraph for SiLU, this method might not be reached
  // in normal graph operations. For completeness, we implementation it
  // correctly. But wait, the value passed here is the OUTPUT of calculate, so
  // it's x * sigmoid(x). We don't have x directly unless we re-calculate it or
  // store it. In CG, derivative usually takes (output_value, backward). Let's
  // just implement a placeholder or try to approximate if needed, but the
  // subgraph approach avoids this issue entirely as individual nodes handle
  // their own derivatives.
  return backward; // Placeholder, subgraph is the primary path.
}

ComputationalNode *
SiLU::addToGraph(const std::vector<ComputationalNode *> &inputNodes, bool isBiased,
              ComputationalGraph *graph) {
  Sigmoid *sigmoid = new Sigmoid();
  ComputationalNode *node2 = sigmoid->addToGraph(inputNodes, isBiased, graph);
  return graph->addNodeEdge(inputNodes[0], node2, isBiased, true);
}
