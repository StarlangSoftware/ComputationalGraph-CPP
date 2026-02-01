//
// Created by Olcay YILDIZ on 1.02.2026.
//

#ifndef COMPUTATIONALGRAPH_DEEPNETWORK_H
#define COMPUTATIONALGRAPH_DEEPNETWORK_H
#include "NeuralNetwork.h"


class DeepNetwork : public NeuralNetwork{
public:
    DeepNetwork();
    void train(vector<Tensor> trainSet, NeuralNetworkParameter parameters) override;
};


#endif //COMPUTATIONALGRAPH_DEEPNETWORK_H