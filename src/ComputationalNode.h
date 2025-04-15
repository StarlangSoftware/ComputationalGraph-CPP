//
// Created by Olcay Taner YILDIZ on 10.04.2025.
//

// computational_node.h
#ifndef COMPUTATIONAL_NODE_H
#define COMPUTATIONAL_NODE_H

#include <Matrix.h>
#include <optional>

#include "FunctionType.h"

namespace ComputationalGraph {

    class ComputationalNode {
    private:
        std::optional<FunctionType> functionType;
        char operatorSymbol;
        Matrix value;
        Matrix backward;
        bool isLearnable;
        bool isBiased;

    public:
        ComputationalNode(bool learnable, FunctionType functionType, bool isBiased);
        ComputationalNode(bool learnable, char operatorSymbol, bool isBiased);
        ComputationalNode(const Matrix& value, char operatorSymbol);

        bool getIsBiased() const;
        std::optional<FunctionType> getFunctionType() const;
        char getOperator() const;
        const Matrix& getValue() const;
        bool getIsLearnable() const;
        const Matrix& getBackward() const;

        void setValue(const Matrix& newValue);
        void setBackward(const Matrix& newBackward);

        void updateValue();
    };

} // namespace ComputationalGraph

#endif // COMPUTATIONAL_NODE_H