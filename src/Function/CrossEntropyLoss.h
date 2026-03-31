//
// Created by Olcay YILDIZ on 26.01.2026.
//

#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include "Function.h"
#include <vector>

class CrossEntropyLoss : public Function {
public:
  CrossEntropyLoss();
  [[nodiscard]] Tensor calculate(const Tensor &value) override;
  [[nodiscard]] Tensor derivative(const Tensor &value,
                                  const Tensor &backward) override;
  ComputationalNode *addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                             bool isBiased, ComputationalGraph *graph) override;
};

#endif // CROSS_ENTROPY_LOSS_H
