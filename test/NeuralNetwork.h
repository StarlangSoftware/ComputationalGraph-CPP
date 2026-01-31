//
// Created by Olcay Taner YILDIZ on 30.01.2026.
//

#ifndef COMPUTATIONALGRAPH_NEURALNETWORK_H
#define COMPUTATIONALGRAPH_NEURALNETWORK_H
#include "../src/ComputationalGraph.h"


class NeuralNetwork : public ComputationalGraph{
protected:
    Tensor createInputTensor(const Tensor& instance);
};


#endif //COMPUTATIONALGRAPH_NEURALNETWORK_H