//
// Created by Olcay YILDIZ on 28.01.2026.
//

#ifndef COMPUTATIONALGRAPH_ADAM_H
#define COMPUTATIONALGRAPH_ADAM_H
#include "SGDMomentum.h"


class Adam : public SGDMomentum{
public:
    Adam(double learningRate, double etaDecrease, double beta1, double beta2, double epsilon);
    void updateValues(map<ComputationalNode*, vector<ComputationalNode*>>& nodeMap);
protected:
    vector<double> calculate(const ComputationalNode& node);
    void setGradients(ComputationalNode &node) override;
private:
    map<ComputationalNode, vector<double>> momentumMap;
    double beta2;
    double epsilon;
    double currentBeta1;
    double currentBeta2;
};


#endif //COMPUTATIONALGRAPH_ADAM_H