//
// Created by Olcay YILDIZ on 27.01.2026.
//

#include "Optimizer.h"

Optimizer::Optimizer(const double learningRate, const double etaDecrease) {
    this->learningRate = learningRate;
    this->etaDecrease = etaDecrease;
}

/**
 * Updates the learning rate of the optimizer.
 */
void Optimizer::setLearningRate() {
    this->learningRate *= this->etaDecrease;
}

/**
 * Checks if broadcasting be applied to the corresponding node.
 * @param node The node to check.
 * @return The index of the dimension where broadcasting is to be applied. -1 if broadcasting is not to be applied.
 */
int Optimizer::broadcast(const ComputationalNode& node) {
    vector<int> v = node.getValue().getShape();
    vector<int> b = node.getBackward().getShape();
    int index = -1;
    for (int i = 0; i < v.size(); i++) {
        if (v[i] != b[i]) {
            if (v[i] == 1) {
                if (index != -1) {
                    return -1;
                }
                index = i;
            }
        }
    }
    return index;
}

/**
 * Recursive helper function to update the values of learnable nodes.
 * @param visited A set of visited nodes.
 * @param node The current node being processed.
 * @param nodeMap A map of nodes to their children.
 */
void Optimizer::updateRecursive(set<ComputationalNode*>& visited, ComputationalNode* node,
    const map<ComputationalNode*, vector<ComputationalNode*>>& nodeMap) {
    visited.emplace(node);
    if (node->isLearnable()) {
        int index = broadcast(*node);
        if (index != -1) {
            int v = 1, b = 1;
            for (int i = node->getValue().getShape().size() - 1; i >= index; i--) {
                v *= node->getValue().getShape()[i];
                b *= node->getBackward().getShape()[i];
            }
            vector<double> backwardValues = node->getBackward().getData();
            auto *values = new double[node->getValue().getData().size()];
            for (int i = 0; i < backwardValues.size(); i++) {
                for (int j = i; j < i + b; j++) {
                    values[((j - i) % v) + v * (j / b)] += backwardValues[j];
                }
                i += b - 1;
            }
            vector<double> list;
            for (int i = 0; i < node->getValue().getData().size(); i++) {
                list.push_back(values[i]);
            }
            delete[] values;
            node->setBackward(Tensor(list, node->getValue().getShape()));
        }
        this->setGradients(*node);
        node->updateValue();
    }
    if (nodeMap.contains(node)) {
        for (ComputationalNode* child: nodeMap.at(node)) {
            if (!visited.contains(child)) {
                updateRecursive(visited, child, nodeMap);
            }
        }
    }
}

/**
 * Updates the values of all learnable nodes in the graph.
 * @param nodeMap A map of nodes to their children.
 */
void Optimizer::updateValues(map<ComputationalNode*, vector<ComputationalNode*>> &nodeMap) {
    set<ComputationalNode*> visited;
    vector<ComputationalNode*> nodes;
    nodes.reserve(nodeMap.size());
    for (auto &[node, value]: nodeMap) {
        nodes.emplace_back(node);
    }
    for (ComputationalNode* node : nodes) {
        if (!visited.contains(node)) {
            updateRecursive(visited, node, nodeMap);
        }
    }
}
