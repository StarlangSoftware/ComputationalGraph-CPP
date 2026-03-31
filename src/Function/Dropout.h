//
// Created by Olcay YILDIZ on 26.01.2026.
//

#ifndef COMPUTATIONALGRAPH_DROPOUT_H
#define COMPUTATIONALGRAPH_DROPOUT_H

#include "Function.h"
#include <random>
#include <vector>

class Dropout : public Function {
private:
  double p;
  std::default_random_engine randomEngine;
  std::vector<double> mask;

public:
  Dropout(double p, const std::default_random_engine &randomEngine);
  [[nodiscard]] Tensor calculate(const Tensor &value) override;
  [[nodiscard]] Tensor derivative(const Tensor &value,
                                  const Tensor &backward) override;
  ComputationalNode *addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                             bool isBiased, ComputationalGraph *graph) override;
};

#endif // COMPUTATIONALGRAPH_DROPOUT_H