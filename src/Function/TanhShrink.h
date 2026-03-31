//
// Created by Olcay YILDIZ on 26.01.2026.
//

#ifndef TANH_SHRINK_H
#define TANH_SHRINK_H

#include "Function.h"
#include <vector>

class TanhShrink : public Function {
public:
  TanhShrink();
  [[nodiscard]] Tensor calculate(const Tensor &value) override;
  [[nodiscard]] Tensor derivative(const Tensor &value,
                                  const Tensor &backward) override;
  ComputationalNode *addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                             bool isBiased, ComputationalGraph *graph) override;
};

#endif // TANH_SHRINK_H
