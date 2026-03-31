//
// Created by Olcay YILDIZ on 26.01.2026.
//

#ifndef POWER_H
#define POWER_H

#include "Function.h"
#include <vector>

class Power : public Function {
private:
  double exponent;

public:
  explicit Power(double exponent);
  [[nodiscard]] Tensor calculate(const Tensor &value) override;
  [[nodiscard]] Tensor derivative(const Tensor &value,
                                  const Tensor &backward) override;
  ComputationalNode *addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                             bool isBiased, ComputationalGraph *graph) override;
};

#endif // POWER_H
