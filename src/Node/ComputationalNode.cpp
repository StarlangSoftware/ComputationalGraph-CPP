//
// Created by Olcay YILDIZ on 27.01.2026.
//

#include "ComputationalNode.h"

#include <utility>

/**
 * Initializes a ComputationalNode.
 * @param learnable Indicates whether the node is learnable (e.g., weights)
 * @param isBiased Indicates whether the node is biased
 * @param function The function (e.g., activation like SIGMOID)
 * @param value The tensor value associated with the node (optional)
 */
ComputationalNode::ComputationalNode(bool learnable, bool isBiased, Function* function, Tensor value) {
    this->nodeType = COMPUTATIONAL_NODE_TYPE;
    this->value = std::move(value);
    this->valueNull = false;
    this->learnable = learnable;
    this->biased = isBiased;
    this->function = function;
}

/**
 * Constructor overload for function type initialization
 * @param learnable Indicates whether the node is learnable (e.g., weights)
 * @param isBiased Indicates whether the node is biased
 * @param function The function (e.g., activation like SIGMOID)
 */
ComputationalNode::ComputationalNode(bool learnable, Function *function, bool isBiased) : ComputationalNode(learnable, isBiased, function, Tensor({0})) {
    this->valueNull = true;
}

/**
 * Constructor overload for operator initialization
 * @param learnable Indicates whether the node is learnable (e.g., weights)
 * @param isBiased Indicates whether the node is biased
 */
ComputationalNode::ComputationalNode(bool learnable, bool isBiased) : ComputationalNode(learnable, isBiased, nullptr, Tensor({0})) {
    this->valueNull = true;
}

bool ComputationalNode::isBiased() const{
    return this->biased;
}

Function* ComputationalNode::getFunction() const {
    return this->function;
}

Tensor ComputationalNode::getValue() const {
    return this->value;
}

void ComputationalNode::setValue(const Tensor& _value) {
    this->value = _value;
    this->valueNull = false;
}

void ComputationalNode::updateValue() {
    this->setValue(this->value.add(backward));
}

bool ComputationalNode::isLearnable() const {
    return this->learnable;
}

Tensor ComputationalNode::getBackward() const {
    return this->backward;
}

void ComputationalNode::setBackward(const Tensor &_backward) {
    this->backward = _backward;
    this->backwardNull = false;
}

bool ComputationalNode::isValueNull() const {
    return this->valueNull;
}

void ComputationalNode::setValueNull() {
    this->value = Tensor({0});
    this->valueNull = true;
}

void ComputationalNode::setBackwardNull() {
    this->backward = Tensor({0});
    this->backwardNull = true;
}

enum NodeType ComputationalNode::getNodeType() const {
    return this->nodeType;
}

bool ComputationalNode::isBackwardNull() const {
    return this->backwardNull;
}
