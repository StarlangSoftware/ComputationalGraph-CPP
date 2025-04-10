#ifndef COMPUTATIONAL_GRAPH_WITH_TENSOR_HPP
#define COMPUTATIONAL_GRAPH_WITH_TENSOR_HPP

#include <unordered_map>
#include <vector>
#include <list>

#include "Tensor.h"
#include "ComputationalNode.cpp"

namespace ComputationalGraphWithTensor {

    class ComputationalGraphWithTensor {
    private:
        std::unordered_map<std::shared_ptr<ComputationalGraph::ComputationalNode>, std::vector<std::shared_ptr<ComputationalGraph::ComputationalNode>>> nodeMap;
        std::unordered_map<std::shared_ptr<ComputationalGraph::ComputationalNode>, std::vector<std::shared_ptr<ComputationalNode>>> reverseNodeMap;

    public:
        ComputationalGraphWithTensor() = default;

        std::shared_ptr<ComputationalNode> addEdge(std::shared_ptr<ComputationalNode> first, std::shared_ptr<ComputationalNode> second, bool isBiased) {
            auto newNode = std::make_shared<ComputationalNode>(false, second->getOperator(), isBiased);
            nodeMap[first].push_back(newNode);
            nodeMap[second].push_back(newNode);
            reverseNodeMap[newNode] = { first, second };
            return newNode;
        }

        std::shared_ptr<ComputationalNode> addEdge(std::shared_ptr<ComputationalNode> node, FunctionType type, bool isBiased) {
            auto newNode = std::make_shared<ComputationalNode>(false, type, isBiased);
            nodeMap[node].push_back(newNode);
            reverseNodeMap[newNode] = { node };
            return newNode;
        }

        std::list<std::shared_ptr<ComputationalNode>> topologicalSort() {
            std::list<std::shared_ptr<ComputationalNode>> sortedNodes;
            std::unordered_set<std::shared_ptr<ComputationalNode>> visited;

            for (const auto& pair : nodeMap) {
                if (visited.find(pair.first) == visited.end()) {
                    sort(pair.first, visited, sortedNodes);
                }
            }
            return sortedNodes;
        }

    private:
        void sort(std::shared_ptr<ComputationalNode> node, std::unordered_set<std::shared_ptr<ComputationalNode>>& visited,
                  std::list<std::shared_ptr<ComputationalNode>>& sortedNodes) {
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

    public:
        void updateValues() {
            std::unordered_set<std::shared_ptr<ComputationalNode>> visited;
            for (const auto& pair : nodeMap) {
                if (visited.find(pair.first) == visited.end()) {
                    update(pair.first, visited);
                }
            }
        }

    private:
        void update(std::shared_ptr<ComputationalNode> node, std::unordered_set<std::shared_ptr<ComputationalNode>>& visited) {
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

    public:
        std::shared_ptr<Tensor> calculateDerivative(std::shared_ptr<ComputationalNode> node, std::shared_ptr<ComputationalNode> child) {
            auto left = reverseNodeMap[child][0];

            if (reverseNodeMap[child].size() == 1) {
                switch (child->getFunctionType().value()) {
                    case FunctionType::SIGMOID:
                        return std::make_shared<Tensor>(child->getBackward()->elementwise_op(child->getValue()->sigmoid_derivative(), std::multiplies<float>()));
                    case FunctionType::TANH:
                        return std::make_shared<Tensor>(child->getBackward()->elementwise_op(child->getValue()->tanh_derivative(), std::multiplies<float>()));
                    case FunctionType::RELU:
                        return std::make_shared<Tensor>(child->getBackward()->elementwise_op(child->getValue()->relu_derivative(), std::multiplies<float>()));
                    case FunctionType::SOFTMAX:
                        return std::make_shared<Tensor>(child->getBackward()->elementwise_op(child->getValue()->softmax_derivative(), std::multiplies<float>()));
                    default:
                        return nullptr;
                }
            } else {
                auto right = reverseNodeMap[child][1];
                switch (child->getOperator()) {
                    case '*':
                        return left == node ?
                               std::make_shared<Tensor>(child->getBackward()->dot(*right->getValue())) :
                               std::make_shared<Tensor>(left->getValue()->transpose().dot(*child->getBackward()));
                    case '+':
                        return std::make_shared<Tensor>(*child->getBackward());
                    case '-':
                        return left == node ?
                               std::make_shared<Tensor>(*child->getBackward()) :
                               std::make_shared<Tensor>(child->getBackward()->negate());
                }
            }
            return nullptr;
        }

        std::vector<int> forwardCalculation() {
            auto sortedNodes = topologicalSort();
            auto output = sortedNodes.front();

            while (sortedNodes.size() > 1) {
                auto currentNode = sortedNodes.back();
                sortedNodes.pop_back();

                for (auto& child : nodeMap[currentNode]) {
                    if (!child->hasValue()) {
                        if (child->getFunctionType()) {
                            switch (child->getFunctionType().value()) {
                                case FunctionType::TANH: child->setValue(std::make_shared<Tensor>(currentNode->getValue()->tanh())); break;
                                case FunctionType::SIGMOID: child->setValue(std::make_shared<Tensor>(currentNode->getValue()->sigmoid())); break;
                                case FunctionType::RELU: child->setValue(std::make_shared<Tensor>(currentNode->getValue()->relu())); break;
                                case FunctionType::SOFTMAX: child->setValue(std::make_shared<Tensor>(currentNode->getValue()->softmax())); break;
                                default: break;
                            }
                        } else {
                            child->setValue(std::make_shared<Tensor>(*currentNode->getValue()));
                        }
                    } else {
                        switch (child->getOperator()) {
                            case '*': child->setValue(std::make_shared<Tensor>(child->getValue()->dot(*currentNode->getValue()))); break;
                            case '+': child->setValue(std::make_shared<Tensor>(child->getValue()->elementwise_op(*currentNode->getValue(), std::plus<float>()))); break;
                            case '-': child->setValue(std::make_shared<Tensor>(child->getValue()->elementwise_op(*currentNode->getValue(), std::minus<float>()))); break;
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
    };

} // namespace ComputationalGraph

#endif // COMPUTATIONAL_GRAPH_HPP
