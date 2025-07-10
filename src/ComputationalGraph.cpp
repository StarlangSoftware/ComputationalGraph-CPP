#include "ComputationalGraph.h"

ComputationalGraph::ComputationalGraph() = default;

ComputationalNode* ComputationalGraph::addEdge(ComputationalNode* first, ComputationalNode* second, bool isBiased) {
    auto newNode = new ComputationalNode(false, second->getOperator(), isBiased);
    nodeMap[first].push_back(newNode);
    reverseNodeMap[newNode].push_back(first);

    nodeMap[second].push_back(newNode);
    reverseNodeMap[newNode].push_back(second);

    return newNode;
}

ComputationalNode* ComputationalGraph::addEdge(ComputationalNode* node, FunctionType type, bool isBiased) {
    auto newNode = new ComputationalNode(false, type, isBiased);
    nodeMap[node].push_back(newNode);
    reverseNodeMap[newNode].push_back(node);
    return newNode;
}

void ComputationalGraph::sort(ComputationalNode* node, std::unordered_set<ComputationalNode*>& visited, std::list<ComputationalNode*>& sortedNodes) {
    visited.insert(node);
    if (nodeMap.find(node) != nodeMap.end()) {
        for (auto child : nodeMap[node]) {
            if (visited.find(child) == visited.end()) {
                sort(child, visited, sortedNodes);
            }
        }
    }
    sortedNodes.push_back(node);
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

void ComputationalGraph::updateRecursive(ComputationalNode* node, std::unordered_set<ComputationalNode*>& visited) {
    visited.insert(node);
    if (!node->isLearnable()) {}
    else {
        node->updateValue();
    }
    if (nodeMap.find(node) != nodeMap.end()) {
        for (auto child : nodeMap[node]) {
            if (visited.find(child) == visited.end()) {
                updateRecursive(child, visited);
            }
        }
    }
}

void ComputationalGraph::updateValues() {
    std::unordered_set<ComputationalNode*> visited;
    for (const auto& pair : nodeMap) {
        if (visited.find(pair.first) == visited.end()) {
            updateRecursive(pair.first, visited);
        }
    }
}

void ComputationalGraph::clearRecursive(ComputationalNode* node, std::unordered_set<ComputationalNode*>& visited) {
    visited.insert(node);
    if (!node->isLearnable()) {
        node->setValue(nullptr);
    }
    node->setBackward(nullptr);

    if (nodeMap.find(node) != nodeMap.end()) {
        for (auto child : nodeMap[node]) {
            if (visited.find(child) == visited.end()) {
                clearRecursive(child, visited);
            }
        }
    }
}

void ComputationalGraph::clear() {
    std::unordered_set<ComputationalNode*> visited;
    for (const auto& pair : nodeMap) {
        if (visited.find(pair.first) == visited.end()) {
            clearRecursive(pair.first, visited);
        }
    }
}

Tensor ComputationalGraph::calculateDerivative(ComputationalNode* node, ComputationalNode* child) {
    auto& parents = reverseNodeMap[child];
    ComputationalNode* left = parents[0];
    if (parents.size() == 1) {
        switch (child->getFunctionType()) {
            case FunctionType::SIGMOID:
                return *(child->getBackward()) * child->getValue()->sigmoid_derivative();
            case FunctionType::TANH:
                return *(child->getBackward()) * child->getValue()->tanh_derivative();
            case FunctionType::RELU:
                return *(child->getBackward()) * child->getValue()->relu_derivative();
            case FunctionType::SOFTMAX:
                return *(child->getBackward()) * child->getValue()->softmax_derivative();
            default:
                throw std::invalid_argument("Unsupported function type in calculateDerivative");
        }
    } else {
        ComputationalNode* right = parents[1];
        switch (child->getOperator()) {
            case '*':
                if (left == node) {
                    if (!child->isBiased()) {
                        return child->getBackward()->dot(right->getValue()->transpose());
                    }
                    // If child is biased, remove last column (as in Python)
                    return child->getBackward()->partial({0, 0}, {child->getBackward()->getRow(), child->getBackward()->getColumn() - 1})
                            .dot(right->getValue()->transpose());
                } else {
                    return left->getValue()->transpose().dot(*(child->getBackward()));
                }
            case '+':
                return *(child->getBackward());
            case '-':
                if (left == node) {
                    return *(child->getBackward());
                } else {
                    return child->getBackward()->negate();
                }
            default:
                throw std::invalid_argument("Unsupported operator in calculateDerivative");
        }
    }
}

void ComputationalGraph::getBiased(ComputationalNode* node) {
    Tensor* oldVal = node->getValue();
    Tensor biased(oldVal->getRow(), oldVal->getColumn() + 1);
    for (int i = 0; i < oldVal->getRow(); ++i) {
        for (int j = 0; j < oldVal->getColumn(); ++j) {
            biased.set({i, j}, oldVal->get({i, j}));
        }
        biased.set({i, oldVal->getColumn()}, 1.0f);
    }
    node->setValue(new Tensor(biased));
}

std::vector<int> ComputationalGraph::forwardCalculation() {
    std::list<ComputationalNode*> sortedNodes = topologicalSort();
    auto outputNode = sortedNodes.front();

    // Work from leaves to output, as in the Python logic.
    while (sortedNodes.size() > 1) {
        ComputationalNode* currentNode = sortedNodes.back();
        sortedNodes.pop_back();

        for (auto child : nodeMap[currentNode]) {
            if (!child->hasValue()) {
                if (child->getFunctionType() != FunctionType::NONE) {
                    switch (child->getFunctionType()) {
                        case FunctionType::SIGMOID:
                            child->setValue(new Tensor(currentNode->getValue()->sigmoid()));
                            break;
                        case FunctionType::TANH:
                            child->setValue(new Tensor(currentNode->getValue()->tanh()));
                            break;
                        case FunctionType::RELU:
                            child->setValue(new Tensor(currentNode->getValue()->relu()));
                            break;
                        case FunctionType::SOFTMAX:
                            child->setValue(new Tensor(currentNode->getValue()->softmax()));
                            break;
                        default:
                            throw std::invalid_argument("Unsupported function type in forwardCalculation");
                    }
                } else {
                    if (currentNode->isBiased()) {
                        getBiased(currentNode);
                    }
                    child->setValue(new Tensor(*currentNode->getValue()));
                }
            } else {
                // Binary operations, shape checks should be managed in Tensor
                switch (child->getOperator()) {
                    case '*':
                        if (currentNode->isBiased()) {
                            getBiased(currentNode);
                        }
                        if (child->getValue()->getColumn() == currentNode->getValue()->getRow()) {
                            child->setValue(new Tensor(child->getValue()->dot(*currentNode->getValue())));
                        } else {
                            child->setValue(new Tensor(currentNode->getValue()->dot(*child->getValue())));
                        }
                        break;
                    case '+':
                        child->setValue(new Tensor(child->getValue()->elementwise_op(*currentNode->getValue(), std::plus<float>())));
                        break;
                    case '-':
                        child->setValue(new Tensor(child->getValue()->elementwise_op(*currentNode->getValue(), std::minus<float>())));
                        break;
                    default:
                        throw std::invalid_argument("Unsupported operator in forwardCalculation");
                }
            }
        }
    }

    std::vector<int> classLabelIndices;
    for (int i = 0; i < outputNode->getValue()->getRow(); ++i) {
        float max_val = -1e9f;
        int label_index = -1;
        for (int j = 0; j < outputNode->getValue()->getColumn(); ++j) {
            float val = outputNode->getValue()->get({i, j});
            if (val > max_val) {
                max_val = val;
                label_index = j;
            }
        }
        classLabelIndices.push_back(label_index);
    }
    return classLabelIndices;
}

std::vector<int> ComputationalGraph::predict() {
    std::vector<int> result = forwardCalculation();
    clear();
    return result;
}
