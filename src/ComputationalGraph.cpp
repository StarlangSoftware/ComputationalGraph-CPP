
#include "ComputationalGraph.h"


ComputationalNode* ComputationalGraph::addEdge(ComputationalNode* first, ComputationalNode* second, bool isBiased) {
    auto newNode = new ComputationalNode(false, second->getOperator(), isBiased);
    nodeMap[first].push_back(newNode);
    nodeMap[second].push_back(newNode);
    reverseNodeMap[newNode] = {first, second};
    return newNode;
}

ComputationalNode* ComputationalGraph::addEdge(ComputationalNode* node, FunctionType type, bool isBiased) {
    auto newNode = new ComputationalNode(false, type, isBiased);
    nodeMap[node].push_back(newNode);
    reverseNodeMap[newNode] = {node};
    return newNode;
}

std::list<ComputationalNode *> ComputationalGraph::topologicalSort() {
    std::list<ComputationalNode *> sortedNodes;
    std::unordered_set<ComputationalNode *> visited;

    for (const auto &pair: nodeMap) {
        if (visited.find(pair.first) == visited.end()) {
            sort(pair.first, visited, sortedNodes);
        }
    }
    return sortedNodes;
}

void ComputationalGraph::sort(ComputationalNode *node,
                                                  std::unordered_set<ComputationalNode *> &visited,
                                                  std::list<ComputationalNode *> &sortedNodes) {
    visited.insert(node);
    if (nodeMap.find(node) != nodeMap.end()) {
        for (auto &child: nodeMap[node]) {
            if (visited.find(child) == visited.end()) {
                sort(child, visited, sortedNodes);
            }
        }
    }
    sortedNodes.push_back(node);
}

void ComputationalGraph::updateValues() {
    std::unordered_set<ComputationalNode *> visited;
    for (const auto &pair: nodeMap) {
        if (visited.find(pair.first) == visited.end()) {
            update(pair.first, visited);
        }
    }
}

void ComputationalGraph::update(ComputationalNode *node, std::unordered_set<ComputationalNode *> &visited) {
    visited.insert(node);
    if (node->getIsLearnable()) {
        node->updateValue();
    }
    if (nodeMap.find(node) != nodeMap.end()) {
        for (auto &child: nodeMap[node]) {
            if (visited.find(child) == visited.end()) {
                update(child, visited);
            }
        }
    }
}

Tensor ComputationalGraph::calculateDerivative(ComputationalNode *node, ComputationalNode *child) {
    auto left = reverseNodeMap[child][0];

    if (reverseNodeMap[child].size() == 1) {
        switch (child->getFunctionType().value()) {
            case FunctionType::SIGMOID:
                return child->getBackward()->elementwise_op(child->getValue()->sigmoid_derivative(),
                                                            std::multiplies<float>());
            case FunctionType::TANH:
                return child->getBackward()->elementwise_op(child->getValue()->tanh_derivative(),
                                                            std::multiplies<float>());
            case FunctionType::RELU:
                return child->getBackward()->elementwise_op(child->getValue()->relu_derivative(),
                                                            std::multiplies<float>());
            case FunctionType::SOFTMAX:
                return child->getBackward()->elementwise_op(child->getValue()->softmax_derivative(),
                                                            std::multiplies<float>());
            default:
                return Tensor();
        }
    } else {
        auto right = reverseNodeMap[child][1];
        switch (child->getOperator()) {
            case '*':
                return (left == node) ?
                       child->getBackward()->dot(*right->getValue()) :
                       left->getValue()->transpose().dot(*child->getBackward());
            case '+':
                return *child->getBackward();
            case '-':
                return (left == node) ? *child->getBackward() : child->getBackward()->negate();
        }
    }
    return Tensor();
}

std::vector<int> ComputationalGraph::forwardCalculation() {
    auto sortedNodes = topologicalSort();
    auto output = sortedNodes.front();

    while (sortedNodes.size() > 1) {
        auto currentNode = sortedNodes.back();
        sortedNodes.pop_back();

        for (auto &child: nodeMap[currentNode]) {
            if (!child->hasValue()) {
                if (child->getFunctionType()) {
                    switch (child->getFunctionType().value()) {
                        case FunctionType::TANH:
                            child->setValue(new Tensor(currentNode->getValue()->tanh()));
                            break;
                        case FunctionType::SIGMOID:
                            child->setValue(new Tensor(currentNode->getValue()->sigmoid()));
                            break;
                        case FunctionType::RELU:
                            child->setValue(new Tensor(currentNode->getValue()->relu()));
                            break;
                        case FunctionType::SOFTMAX:
                            child->setValue(new Tensor(currentNode->getValue()->softmax()));
                            break;
                        default:
                            break;
                    }
                } else {
                    child->setValue(new Tensor(*currentNode->getValue()));
                }
            } else {
                switch (child->getOperator()) {
                    case '*':
                        child->setValue(new Tensor(child->getValue()->dot(*currentNode->getValue())));
                        break;
                    case '+':
                        child->setValue(new Tensor(
                                child->getValue()->elementwise_op(*currentNode->getValue(), std::plus<float>())));
                        break;
                    case '-':
                        child->setValue(new Tensor(
                                child->getValue()->elementwise_op(*currentNode->getValue(), std::minus<float>())));
                        break;
                }
            }
        }
    }

    std::vector<int> classLabelIndex;
    for (size_t i = 0; i < output->getValue()->getRow(); i++) {
        float max = -1e9;
        int labelIndex = -1;
        for (size_t j = 0; j < output->getValue()->getColumn(); j++) {
            if (output->getValue()->get({static_cast<int>(i), static_cast<int>(j)}) > max) {
                max = output->getValue()->get({static_cast<int>(i), static_cast<int>(j)});
                labelIndex = static_cast<int>(j);
            }
        }
        classLabelIndex.push_back(labelIndex);
    }
    return classLabelIndex;
}
