#ifndef COMPUTATIONAL_GRAPH_HPP
#define COMPUTATIONAL_GRAPH_HPP

#include <unordered_map>
#include <vector>
#include <list>
#include <memory>
#include <optional>
#include "ComputationalNode.hpp"
#include "FunctionType.hpp"
#include "math/Matrix.hpp"
#include "Sigmoid.hpp"
#include "Tanh.hpp"
#include "ReLU.hpp"
#include "Softmax.hpp"

namespace ComputationalGraph {

    class ComputationalGraph {
    private:
        std::unordered_map<std::shared_ptr<ComputationalNode>, std::vector<std::shared_ptr<ComputationalNode>>> nodeMap;
        std::unordered_map<std::shared_ptr<ComputationalNode>, std::vector<std::shared_ptr<ComputationalNode>>> reverseNodeMap;

    public:
        ComputationalGraph() = default;

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
        std::shared_ptr<math::Matrix> calculateDerivative(std::shared_ptr<ComputationalNode> node, std::shared_ptr<ComputationalNode> child) {
            auto left = reverseNodeMap[child][0];

            if (reverseNodeMap[child].size() == 1) {
                std::unique_ptr<math::Function> function;
                switch (child->getFunctionType().value()) {
                    case FunctionType::SIGMOID:
                        function = std::make_unique<math::Sigmoid>();
                        break;
                    case FunctionType::TANH:
                        function = std::make_unique<math::Tanh>();
                        break;
                    case FunctionType::RELU:
                        function = std::make_unique<math::ReLU>();
                        break;
                    case FunctionType::SOFTMAX:
                        function = std::make_unique<math::Softmax>();
                        break;
                    default:
                        return nullptr;
                }
                return std::make_shared<math::Matrix>(child->getBackward()->elementProduct(function->derivative(child->getValue())));
            } else {
                auto right = reverseNodeMap[child][1];
                switch (child->getOperator()) {
                    case '*':
                        return left == node ?
                               std::make_shared<math::Matrix>(child->getBackward()->multiply(right->getValue()->transpose())) :
                               std::make_shared<math::Matrix>(left->getValue()->transpose()->multiply(child->getBackward()));
                    case '+':
                        return std::make_shared<math::Matrix>(*child->getBackward());
                    case '-':
                        return left == node ?
                               std::make_shared<math::Matrix>(*child->getBackward()) :
                               std::make_shared<math::Matrix>(child->getBackward()->negate());
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
                            std::unique_ptr<math::Function> function;
                            switch (child->getFunctionType().value()) {
                                case FunctionType::TANH: function = std::make_unique<math::Tanh>(); break;
                                case FunctionType::SIGMOID: function = std::make_unique<math::Sigmoid>(); break;
                                case FunctionType::RELU: function = std::make_unique<math::ReLU>(); break;
                                case FunctionType::SOFTMAX: function = std::make_unique<math::Softmax>(); break;
                                default: break;
                            }
                            child->setValue(function->calculate(currentNode->getValue()));
                        } else {
                            child->setValue(std::make_shared<math::Matrix>(*currentNode->getValue()));
                        }
                    } else {
                        switch (child->getOperator()) {
                            case '*': child->setValue(child->getValue()->multiply(currentNode->getValue())); break;
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
                        max = output->getValue()->getValue(i, j);
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
