// computational_node.cpp
#include "computational_node.h"

namespace ComputationalGraph {

    ComputationalNode::ComputationalNode(bool learnable, FunctionType functionType, bool isBiased)
            : isLearnable(learnable), functionType(functionType), isBiased(isBiased), operatorSymbol('\0') {}

    ComputationalNode::ComputationalNode(bool learnable, char operatorSymbol, bool isBiased)
            : isLearnable(learnable), operatorSymbol(operatorSymbol), isBiased(isBiased), functionType(std::nullopt) {}

    ComputationalNode::ComputationalNode(const Matrix& value, char operatorSymbol)
            : value(value), isLearnable(true), operatorSymbol(operatorSymbol), functionType(std::nullopt), isBiased(false) {}

    bool ComputationalNode::getIsBiased() const {
        return isBiased;
    }

    std::optional<FunctionType> ComputationalNode::getFunctionType() const {
        return functionType;
    }

    char ComputationalNode::getOperator() const {
        return operatorSymbol;
    }

    const Matrix& ComputationalNode::getValue() const {
        return value;
    }

    bool ComputationalNode::getIsLearnable() const {
        return isLearnable;
    }

    const Matrix& ComputationalNode::getBackward() const {
        return backward;
    }

    void ComputationalNode::setValue(const Matrix& newValue) {
        value = newValue;
    }

    void ComputationalNode::setBackward(const Matrix& newBackward) {
        backward = newBackward;
    }

    void ComputationalNode::updateValue() {
        for (size_t i = 0; i < value.getRow(); ++i) {
            for (size_t j = 0; j < value.getColumn(); ++j) {
                value.setValue(i, j, value.getValue(i, j) + backward.getValue(i, j));
            }
        }
    }

} // namespace ComputationalGraph
