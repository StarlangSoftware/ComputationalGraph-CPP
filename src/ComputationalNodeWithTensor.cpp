#include "ComputationalNodeWithTensor.h"

ComputationalNode::ComputationalNode(bool learnable, FunctionType functionType, bool isBiased)
        : isLearnable(learnable), functionType(functionType), isBiased(isBiased), operatorSymbol('\0') {}

ComputationalNode::ComputationalNode(bool learnable, char operatorSymbol, bool isBiased)
        : isLearnable(learnable), operatorSymbol(operatorSymbol), isBiased(isBiased), functionType(std::nullopt) {}

ComputationalNode::ComputationalNode(const Tensor &value, char operatorSymbol)
        : value(value), isLearnable(true), operatorSymbol(operatorSymbol), functionType(std::nullopt),
          isBiased(false) {}

bool ComputationalNode::getIsBiased() const {
    return isBiased;
}

std::optional<FunctionType> ComputationalNode::getFunctionType() const {
    return functionType;
}

char ComputationalNode::getOperator() const {
    return operatorSymbol;
}

const Tensor &ComputationalNode::getValue() const {
    return value;
}

bool ComputationalNode::getIsLearnable() const {
    return isLearnable;
}

const Tensor &ComputationalNode::getBackward() const {
    return backward;
}

void ComputationalNode::setValue(const Tensor &newValue) {
    value = newValue;
}

void ComputationalNode::setBackward(const Tensor &newBackward) {
    backward = newBackward;
}

void ComputationalNode::updateValue() {
    if (value.getShape() != backward.getShape()) {
        throw std::invalid_argument("Shape mismatch during updateValue()");
    }
    value = value + backward;
}
