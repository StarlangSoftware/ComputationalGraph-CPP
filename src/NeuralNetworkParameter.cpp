//
// Created by Olcay YILDIZ on 28.01.2026.
//

#include "NeuralNetworkParameter.h"

NeuralNetworkParameter::NeuralNetworkParameter(const int seed, const int epoch,
                                               Optimizer *optimizer,
                                               Initialization *initialization,
                                               const double dropout,
                                               Function *lossFunction,
                                               const int batchSize) {
  this->seed = seed;
  this->epoch = epoch;
  this->optimizer = optimizer;
  this->initialization = initialization;
  this->dropout = dropout;
  this->lossFunction = lossFunction;
  this->batchSize = batchSize;
}

NeuralNetworkParameter::~NeuralNetworkParameter() {
  delete optimizer;
  delete initialization;
  delete lossFunction;
}

Optimizer *NeuralNetworkParameter::getOptimizer() const { return optimizer; }

int NeuralNetworkParameter::getEpoch() const { return epoch; }

Initialization *NeuralNetworkParameter::getInitialization() const {
  return initialization;
}

double NeuralNetworkParameter::getDropout() const { return dropout; }

int NeuralNetworkParameter::getSeed() const { return seed; }

Function *NeuralNetworkParameter::getLossFunction() const {
  return lossFunction;
}

int NeuralNetworkParameter::getBatchSize() const { return batchSize; }
