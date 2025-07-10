#ifndef COMPUTATIONALGRAPH_COMPUTATIONALNODEWITHTENSOR_H
#define COMPUTATIONALGRAPH_COMPUTATIONALNODEWITHTENSOR_H

#include <optional>
#include "FunctionType.h"
#include <Tensor.h>

class ComputationalNode {
private:
    std::optional<FunctionType> functionType;
    char operatorSymbol;
    Tensor value;
    Tensor backward;
    bool isBiased;
    bool learnable; // <-- You NEED this line

public:
    ComputationalNode(bool learnable, FunctionType functionType, bool isBiased);
    ComputationalNode(bool learnable, char operatorSymbol, bool isBiased);
    ComputationalNode(const Tensor &value, char operatorSymbol);

    bool isLearnable() const; // Use only this one
    bool getIsBiased() const;
    std::optional<FunctionType> getFunctionType() const;
    char getOperator() const;
    const Tensor& getValue() const;
    const Tensor& getBackward() const;

    void setValue(const Tensor &newValue);
    void setBackward(const Tensor &newBackward);
    void updateValue();
};

#endif //COMPUTATIONALGRAPH_COMPUTATIONALNODEWITHTENSOR_H
