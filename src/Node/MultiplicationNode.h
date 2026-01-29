//
// Created by Olcay YILDIZ on 27.01.2026.
//

#ifndef COMPUTATIONALGRAPH_MULTIPLICATIONNODE_H
#define COMPUTATIONALGRAPH_MULTIPLICATIONNODE_H
#include "ComputationalNode.h"


class MultiplicationNode : public ComputationalNode {
private:
    bool hadamard = false;
    ComputationalNode priorityNode = ComputationalNode(false, false);
    bool priorityNull = true;
public:
    MultiplicationNode(bool learnable, bool isBiased, bool isHadamard);
    MultiplicationNode(bool learnable, bool isBiased, bool isHadamard, ComputationalNode priorityNode);
    MultiplicationNode(bool learnable, bool isBiased, Tensor value, bool isHadamard);
    MultiplicationNode(bool learnable, Tensor value);
    MultiplicationNode(Tensor value);
    MultiplicationNode(bool learnable, bool isBiased);
    [[nodiscard]] bool isHadamard() const;
    [[nodiscard]] ComputationalNode getPriorityNode() const;
    [[nodiscard]] bool isPriorityNull() const;
};


#endif //COMPUTATIONALGRAPH_MULTIPLICATIONNODE_H