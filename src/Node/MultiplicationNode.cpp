//
// Created by Olcay YILDIZ on 27.01.2026.
//

#include "MultiplicationNode.h"

#include <utility>

MultiplicationNode::MultiplicationNode(bool learnable, bool isBiased, bool isHadamard) : ComputationalNode(learnable, isBiased){
    this->hadamard = isHadamard;
    this->nodeType = MULTIPLICATION_NODE_TYPE;
}

MultiplicationNode::MultiplicationNode(bool learnable, bool isBiased, bool isHadamard, ComputationalNode* priorityNode) : ComputationalNode(learnable, isBiased){
    this->hadamard = isHadamard;
    this->priorityNode = priorityNode;
    this->priorityNull = false;
    this->nodeType = MULTIPLICATION_NODE_TYPE;
}

MultiplicationNode::MultiplicationNode(bool learnable, bool isBiased, Tensor value, bool isHadamard) : ComputationalNode(learnable, isBiased){
    this->hadamard = isHadamard;
    this->value = std::move(value);
    this->nodeType = MULTIPLICATION_NODE_TYPE;
}

MultiplicationNode::MultiplicationNode(bool learnable, Tensor value) : ComputationalNode(learnable, false){
    this->value = std::move(value);
    this->hadamard = false;
    this->nodeType = MULTIPLICATION_NODE_TYPE;
}

MultiplicationNode::MultiplicationNode(Tensor value) : ComputationalNode(true, false){
    this->value = std::move(value);
    this->hadamard = false;
    this->nodeType = MULTIPLICATION_NODE_TYPE;
}

MultiplicationNode::MultiplicationNode(bool learnable, bool isBiased) : ComputationalNode(learnable, isBiased){
    this->hadamard = false;
    this->nodeType = MULTIPLICATION_NODE_TYPE;
}

bool MultiplicationNode::isHadamard() const {
    return this->hadamard;
}

ComputationalNode* MultiplicationNode::getPriorityNode() const {
    return this->priorityNode;
}

bool MultiplicationNode::isPriorityNull() const {
    return this->priorityNull;
}
