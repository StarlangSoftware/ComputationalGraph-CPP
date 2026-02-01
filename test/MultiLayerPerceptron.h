//
// Created by Olcay YILDIZ on 1.02.2026.
//

#ifndef COMPUTATIONALGRAPH_MULTILAYERPERCEPTRON_H
#define COMPUTATIONALGRAPH_MULTILAYERPERCEPTRON_H
#include "LinearPerceptron.h"


class MultiLayerPerceptron : public NeuralNetwork{
public:
    MultiLayerPerceptron();
    void train(vector<Tensor> trainSet, NeuralNetworkParameter parameters) override;
};


#endif //COMPUTATIONALGRAPH_MULTILAYERPERCEPTRON_H