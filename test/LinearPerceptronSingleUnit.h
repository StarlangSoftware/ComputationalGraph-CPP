//
// Created by Olcay Taner YILDIZ on 30.01.2026.
//

#ifndef COMPUTATIONALGRAPH_LINEARPERCEPTRONSINGLEUNIT_H
#define COMPUTATIONALGRAPH_LINEARPERCEPTRONSINGLEUNIT_H
#include "NeuralNetwork.h"
#include "../src/ComputationalGraph.h"


class LinearPerceptronSingleUnit : public NeuralNetwork{
public:
    LinearPerceptronSingleUnit();
    void train(vector<Tensor> trainSet, NeuralNetworkParameter parameters) override;
    ClassificationPerformance test(vector<Tensor> testSet) override;
protected:
    vector<int> getClassLabels(ComputationalNode *outputNode) override;
};


#endif //COMPUTATIONALGRAPH_LINEARPERCEPTRONSINGLEUNIT_H