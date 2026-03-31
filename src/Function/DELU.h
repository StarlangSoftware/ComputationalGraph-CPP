//
// Created by Olcay YILDIZ on 26.01.2026.
//

#ifndef COMPUTATIONALGRAPH_DELU_H
#define COMPUTATIONALGRAPH_DELU_H
#include "Function.h"

class DELU : public Function {
private:
  double a;
  double b;
  double xc;

public:
  DELU(double a, double b, double xc);
  DELU();
  [[nodiscard]] Tensor calculate(const Tensor &value) override;
  [[nodiscard]] Tensor derivative(const Tensor &value,
                                  const Tensor &backward) override;
  ComputationalNode *addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                             bool isBiased, ComputationalGraph *graph) override;
};

#endif // COMPUTATIONALGRAPH_DELU_H