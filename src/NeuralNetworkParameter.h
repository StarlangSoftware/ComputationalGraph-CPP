//
// Created by Olcay YILDIZ on 28.01.2026.
//

#ifndef COMPUTATIONALGRAPH_NEURALNETWORKPARAMETER_H
#define COMPUTATIONALGRAPH_NEURALNETWORKPARAMETER_H
#include "Function/Function.h"
#include "Initialization/Initialization.h"
#include "Initialization/RandomInitialization.h"
#include "Optimizer/Optimizer.h"

class NeuralNetworkParameter {
private:
  Optimizer *optimizer;
  int epoch;
  int seed;
  Initialization *initialization;
  double dropout;
  Function *lossFunction;
  int batchSize;

public:
  NeuralNetworkParameter(
      int seed, int epoch, Optimizer *optimizer,
      Initialization *initialization = new RandomInitialization(),
      double dropout = 0.0, Function *lossFunction = nullptr,
      int batchSize = 1);
  virtual ~NeuralNetworkParameter();
  [[nodiscard]] Optimizer *getOptimizer() const;
  [[nodiscard]] int getEpoch() const;
  [[nodiscard]] Initialization *getInitialization() const;
  [[nodiscard]] double getDropout() const;
  [[nodiscard]] int getSeed() const;
  [[nodiscard]] Function *getLossFunction() const;
  [[nodiscard]] int getBatchSize() const;
};

#endif // COMPUTATIONALGRAPH_NEURALNETWORKPARAMETER_H
