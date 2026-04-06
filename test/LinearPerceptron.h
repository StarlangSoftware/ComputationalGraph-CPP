//
// Created by Olcay YILDIZ on 31.01.2026.
//

#ifndef COMPUTATIONALGRAPH_LINEARPERCEPTRON_H
#define COMPUTATIONALGRAPH_LINEARPERCEPTRON_H
#include "NeuralNetwork.h"

class LinearPerceptron : public NeuralNetwork {
public:
  LinearPerceptron();
  void train(vector<Tensor> &trainSet,
             NeuralNetworkParameter &parameters) override;
};

#endif // COMPUTATIONALGRAPH_LINEARPERCEPTRON_H