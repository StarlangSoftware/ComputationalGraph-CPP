//
// Created by Olcay Taner YILDIZ on 10.04.2025.
//

#ifndef COMPUTATIONAL_GRAPH_WITH_TENSOR_H
#define COMPUTATIONAL_GRAPH_WITH_TENSOR_H

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>

#include "Tensor.h"
#include "ComputationalNode.h"


class ComputationalGraph {
private:
    std::unordered_map<ComputationalNode*, std::vector<ComputationalNode*>> nodeMap;
    std::unordered_map<ComputationalNode*, std::vector<ComputationalNode*>> reverseNodeMap;

    void sort(ComputationalNode* node,
              std::unordered_set<ComputationalNode*> &visited,
              std::list<ComputationalNode*> &sortedNodes);

    void update(ComputationalNode* node,
                std::unordered_set<ComputationalNode*> &visited);

public:
    ComputationalGraph();

    ComputationalNode* addEdge(ComputationalNode* first, ComputationalNode* second, bool isBiased);

    ComputationalNode* addEdge(ComputationalNode* node, FunctionType type, bool isBiased);

    std::list<ComputationalNode* > topologicalSort();

    void updateValues();

    Tensor calculateDerivative(ComputationalNode* node, ComputationalNode* child);

    std::vector<int> forwardCalculation();
};


#endif // COMPUTATIONAL_GRAPH_WITH_TENSOR_H
