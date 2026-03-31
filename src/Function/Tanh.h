//
// Created by Olcay YILDIZ on 26.01.2026.
//

#ifndef TANH_H
#define TANH_H

#include "Function.h"
#include <vector>

class Tanh : public Function {
public:
  [[nodiscard]] Tensor calculate(const Tensor &value) override;
  [[nodiscard]] Tensor derivative(const Tensor &value,
                                  const Tensor &backward) override;
  ComputationalNode *addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                             bool isBiased, ComputationalGraph *graph) override;
};

#endif // TANH_H
