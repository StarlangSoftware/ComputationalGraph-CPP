//
// Created by Olcay YILDIZ on 26.01.2026.
//

#ifndef LOGARITHM_H
#define LOGARITHM_H

#include "Function.h"
#include <vector>

class Logarithm : public Function {
public:
  Logarithm();
  [[nodiscard]] Tensor calculate(const Tensor &value) override;
  [[nodiscard]] Tensor derivative(const Tensor &value,
                                  const Tensor &backward) override;
  ComputationalNode *addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                             bool isBiased, ComputationalGraph *graph) override;
};

#endif // LOGARITHM_H
