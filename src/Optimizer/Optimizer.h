//
// Created by Olcay YILDIZ on 27.01.2026.
//

#ifndef COMPUTATIONALGRAPH_OPTIMIZER_H
#define COMPUTATIONALGRAPH_OPTIMIZER_H
#include <map>
#include <set>

#include "../Node/ComputationalNode.h"

class Optimizer {
protected:
    double learningRate;
    virtual void setGradients(ComputationalNode& node) = 0;
private:
    double etaDecrease;
    int broadcast(const ComputationalNode& node);
    void updateRecursive(set<ComputationalNode*>& visited, ComputationalNode* node, const map<ComputationalNode*, vector<ComputationalNode*>>& nodeMap);
public:
    virtual ~Optimizer() = default;
    Optimizer(double learningRate, double etaDecrease);
    void setLearningRate();
    virtual void updateValues(map<ComputationalNode*, vector<ComputationalNode*>>& nodeMap);
};


#endif //COMPUTATIONALGRAPH_OPTIMIZER_H