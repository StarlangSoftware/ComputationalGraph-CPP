//
// Created by Olcay YILDIZ on 26.01.2026.
//

#ifndef RELU_H
#define RELU_H

#include "Function.h"
#include <vector>

class ReLU : public Function {
public:
  [[nodiscard]] Tensor calculate(const Tensor &value) override;
  [[nodiscard]] Tensor derivative(const Tensor &value,
                                  const Tensor &backward) override;
  ComputationalNode *addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                             bool isBiased, ComputationalGraph *graph) override;
};

#endif // RELU_H
