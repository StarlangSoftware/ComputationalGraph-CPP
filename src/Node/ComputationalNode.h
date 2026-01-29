//
// Created by Olcay YILDIZ on 27.01.2026.
//

#ifndef COMPUTATIONALGRAPH_COMPUTATIONALNODE_H
#define COMPUTATIONALGRAPH_COMPUTATIONALNODE_H
#include <Tensor.h>
#include "../Function/Function.h"

enum NodeType {
    COMPUTATIONAL_NODE_TYPE,
    CONCATENATED_NODE_TYPE,
    MULTIPLICATION_NODE_TYPE
};

class ComputationalNode {
protected:
    enum NodeType nodeType;
    Tensor value = Tensor({0});
    Tensor backward = Tensor({0});
    bool learnable = false;
    bool biased = false;
    bool valueNull = true;
    bool backwardNull = true;
    Function* function = nullptr;
public:
    ComputationalNode(bool learnable, bool isBiased, Function* function, Tensor value);
    ComputationalNode(bool learnable, Function* function, bool isBiased);
    ComputationalNode(bool learnable, bool isBiased);
    [[nodiscard]] bool isBiased() const;
    [[nodiscard]] Function* getFunction() const;
    [[nodiscard]] Tensor getValue() const;
    void setValue(const Tensor& _value);
    void updateValue();
    [[nodiscard]] bool isLearnable() const;
    [[nodiscard]] Tensor getBackward() const;
    void setBackward(const Tensor& _backward);
    [[nodiscard]] bool isValueNull() const;
    void setValueNull();
    [[nodiscard]] bool isBackwardNull() const;
    void setBackwardNull();
    [[nodiscard]] enum NodeType getNodeType() const;
    bool operator<(const ComputationalNode &computationalNode) const{
        return (value < computationalNode.value) && (backward < computationalNode.backward);
    }
    bool operator==(const ComputationalNode &computationalNode) const{
        if (value.getData().size() != computationalNode.value.getData().size()) {
            return false;
        }
        for (int i = 0; i < computationalNode.value.getData().size(); i++) {
            if (value.getData()[i] != computationalNode.value.getData()[i]) {
                return false;
            }
        }
        return true;
    }
};


#endif //COMPUTATIONALGRAPH_COMPUTATIONALNODE_H