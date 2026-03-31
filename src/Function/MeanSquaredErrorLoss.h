//
// Created by Olcay YILDIZ on 26.01.2026.
//

#ifndef MEAN_SQUARED_ERROR_LOSS_H
#define MEAN_SQUARED_ERROR_LOSS_H

#include "Function.h"
#include <vector>

class MeanSquaredErrorLoss : public Function {
public:
  MeanSquaredErrorLoss();
  [[nodiscard]] Tensor calculate(const Tensor &value) override;
  [[nodiscard]] Tensor derivative(const Tensor &value,
                                  const Tensor &backward) override;
  ComputationalNode *addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                             bool isBiased, ComputationalGraph *graph) override;
};

#endif // MEAN_SQUARED_ERROR_LOSS_H
