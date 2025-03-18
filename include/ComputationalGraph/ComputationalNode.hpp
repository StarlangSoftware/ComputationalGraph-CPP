#ifndef COMPUTATIONAL_NODE_HPP
#define COMPUTATIONAL_NODE_HPP

#include "FunctionType.hpp"
#include "math/Matrix.hpp"
#include <optional>

namespace ComputationalGraph {

    class ComputationalNode {
    private:
        std::optional<FunctionType> functionType;
        char operatorSymbol;
        math::Matrix value;
        math::Matrix backward;
        bool isLearnable;
        bool isBiased;

    public:
        // Constructor for function-based nodes
        ComputationalNode(bool learnable, FunctionType functionType, bool isBiased)
                : isLearnable(learnable), functionType(functionType), isBiased(isBiased), operatorSymbol('\0') {}

        // Constructor for operator-based nodes
        ComputationalNode(bool learnable, char operatorSymbol, bool isBiased)
                : isLearnable(learnable), operatorSymbol(operatorSymbol), isBiased(isBiased), functionType(std::nullopt) {}

        // Constructor with initial value and operator
        ComputationalNode(const math::Matrix& value, char operatorSymbol)
                : value(value), isLearnable(true), operatorSymbol(operatorSymbol), functionType(std::nullopt), isBiased(false) {}

        // Getters
        bool getIsBiased() const { return isBiased; }
        std::optional<FunctionType> getFunctionType() const { return functionType; }
        char getOperator() const { return operatorSymbol; }
        const math::Matrix& getValue() const { return value; }
        bool getIsLearnable() const { return isLearnable; }
        const math::Matrix& getBackward() const { return backward; }

        // Setters
        void setValue(const math::Matrix& newValue) { value = newValue; }
        void setBackward(const math::Matrix& newBackward) { backward = newBackward; }

        // Update value using backward matrix
        void updateValue() {
            for (size_t i = 0; i < value.getRow(); ++i) {
                for (size_t j = 0; j < value.getColumn(); ++j) {
                    value.setValue(i, j, value.getValue(i, j) + backward.getValue(i, j));
                }
            }
        }
    };

} // namespace ComputationalGraph

#endif // COMPUTATIONAL_NODE_HPP
