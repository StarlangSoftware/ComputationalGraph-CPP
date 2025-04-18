#include "ComputationalGraph.h"
#include "ReLU.h"
#include "Sigmoid.h"
#include "ComputationalNode.h"
#include "Softmax.h"
#include "Tanh.h"

ComputationalGraph::ComputationalGraph() = default;

ComputationalNode* ComputationalGraph::addEdge(ComputationalNode* first, ComputationalNode* second, bool isBiased) {
    auto newNode = new ComputationalNode(false, second->getOperator(), isBiased);
    nodeMap[first].push_back(newNode);
    nodeMap[second].push_back(newNode);
    reverseNodeMap[newNode] = { first, second };
    return newNode;
}

ComputationalNode* ComputationalGraph::addEdge(ComputationalNode* node, FunctionType type, bool isBiased) {
    auto newNode = new ComputationalNode(false, type, isBiased);
    nodeMap[node].push_back(newNode);
    reverseNodeMap[newNode] = { node };
    return newNode;
}

std::list<ComputationalNode*> ComputationalGraph::topologicalSort() {
    std::list<ComputationalNode*> sortedNodes;
    std::unordered_set<ComputationalNode*> visited;

    for (const auto& pair : nodeMap) {
        if (visited.find(pair.first) == visited.end()) {
            sort(pair.first, visited, sortedNodes);
        }
    }
    return sortedNodes;
}

void ComputationalGraph::sort(ComputationalNode* node, std::unordered_set<ComputationalNode*>& visited,
                              std::list<ComputationalNode*>& sortedNodes) {
    visited.insert(node);
    if (nodeMap.find(node) != nodeMap.end()) {
        for (auto& child : nodeMap[node]) {
            if (visited.find(child) == visited.end()) {
                sort(child, visited, sortedNodes);
            }
        }
    }
    sortedNodes.push_back(node);
}

void ComputationalGraph::updateValues() {
    std::unordered_set<ComputationalNode*> visited;
    for (const auto& pair : nodeMap) {
        if (visited.find(pair.first) == visited.end()) {
            update(pair.first, visited);
        }
    }
}

void ComputationalGraph::update(ComputationalNode* node, std::unordered_set<ComputationalNode*>& visited) {
    visited.insert(node);
    if (node->isLearnable()) {
        node->updateValue();
    }
    if (nodeMap.find(node) != nodeMap.end()) {
        for (auto& child : nodeMap[node]) {
            if (visited.find(child) == visited.end()) {
                update(child, visited);
            }
        }
    }
}

Matrix* ComputationalGraph::calculateDerivative(ComputationalNode* node, ComputationalNode* child) {
    auto left = reverseNodeMap[child][0];

    if (reverseNodeMap[child].size() == 1) {
        Function* function = nullptr;
        switch (child->getFunctionType().value()) {
            case FunctionType::SIGMOID:
                function = new Sigmoid();
                break;
            case FunctionType::TANH:
                function = new Tanh();
                break;
            case FunctionType::RELU:
                function = new ReLU();
                break;
            case FunctionType::SOFTMAX:
                function = new Softmax();
                break;
            default:
                return nullptr;
        }
        auto result = new Matrix(child->getBackward()->elementProduct(function->derivative(child->getValue())));
        delete function;
        return result;
    } else {
        auto right = reverseNodeMap[child][1];
        switch (child->getOperator()) {
            case '*':
                return left == node ?
                       new Matrix(child->getBackward()->multiply(right->getValue()->transpose())) :
                       new Matrix(left->getValue()->transpose()->multiply(child->getBackward()));
            case '+':
                return new Matrix(*child->getBackward());
            case '-':
                return left == node ?
                       new Matrix(*child->getBackward()) :
                       new Matrix(child->getBackward()->negate());
        }
    }
    return nullptr;
}

std::vector<int> ComputationalGraph::forwardCalculation() {
    auto sortedNodes = topologicalSort();
    auto output = sortedNodes.front();

    while (sortedNodes.size() > 1) {
        auto currentNode = sortedNodes.back();
        sortedNodes.pop_back();

        for (auto& child : nodeMap[currentNode]) {
            if (!child->hasValue()) {
                if (child->getFunctionType()) {
                    Function* function = nullptr;
                    switch (child->getFunctionType().value()) {
                        case FunctionType::TANH: function = new Tanh(); break;
                        case FunctionType::SIGMOID: function = new Sigmoid(); break;
                        case FunctionType::RELU: function = new ReLU(); break;
                        case FunctionType::SOFTMAX: function = new Softmax(); break;
                        default: break;
                    }
                    child->setValue(new Matrix(function->calculate(currentNode->getValue())));
                    delete function;
                } else {
                    child->setValue(new Matrix(*currentNode->getValue()));
                }
            } else {
                switch (child->getOperator()) {
                    case '*': child->setValue(new Matrix(child->getValue()->multiply(currentNode->getValue()))); break;
                    case '+': child->getValue()->add(*currentNode->getValue()); break;
                    case '-': child->getValue()->subtract(*currentNode->getValue()); break;
                }
            }
        }
    }

    std::vector<int> classLabelIndex;
    for (size_t i = 0; i < output->getValue()->getRow(); i++) {
        double max = -1e9;
        int labelIndex = -1;
        for (size_t j = 0; j < output->getValue()->getColumn(); j++) {
            if (output->getValue()->getValue(i, j) > max) {
                max = output->getValue(i, j);
                labelIndex = static_cast<int>(j);
            }
        }
        classLabelIndex.push_back(labelIndex);
    }
    return classLabelIndex;
}
