//
// Created by Olcay YILDIZ on 26.01.2026.
//

#ifndef SILU_H
#define SILU_H

#include "Function.h"
#include <vector>

class SiLU : public Function {
public:
  SiLU();
  [[nodiscard]] Tensor calculate(const Tensor &value) override;
  [[nodiscard]] Tensor derivative(const Tensor &value,
                                  const Tensor &backward) override;
  ComputationalNode *addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                             bool isBiased, ComputationalGraph *graph) override;
};

#endif // SILU_H
