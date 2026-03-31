//
// Created by Olcay YILDIZ on 26.01.2026.
//

#ifndef FUNCTION_H
#define FUNCTION_H

#include "FunctionType.h"
#include "Tensor.h"
#include <vector>

// Forward declarations
class ComputationalNode;
class ComputationalGraph;

class Function {
public:
  FunctionType type;

public:
  virtual ~Function() = default;
  [[nodiscard]] virtual Tensor calculate(const Tensor &value) = 0;
  [[nodiscard]] virtual Tensor derivative(const Tensor &value,
                                          const Tensor &backward) = 0;
  virtual ComputationalNode *
  addToGraph(const std::vector<ComputationalNode *> &inputNodes, bool isBiased,
          ComputationalGraph *graph) = 0;
};

#endif // FUNCTION_H
