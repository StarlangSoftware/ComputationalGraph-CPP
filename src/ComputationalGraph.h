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


template<typename NodeType>
class ComputationalGraph {
private:
    std::unordered_map<NodeType *, std::vector<NodeType *>> nodeMap;
    std::unordered_map<NodeType *, std::vector<NodeType *>> reverseNodeMap;

    void sort(NodeType *node,
              std::unordered_set<NodeType *> &visited,
              std::list<NodeType *> &sortedNodes);

    void update(NodeType *node,
                std::unordered_set<NodeType *> &visited);

public:
    ComputationalGraph();

    NodeType *addEdge(NodeType *first, NodeType *second, bool isBiased);

    NodeType *addEdge(NodeType *node, FunctionType type, bool isBiased);

    std::list<NodeType *> topologicalSort();

    void updateValues();

    Tensor calculateDerivative(NodeType *node, NodeType *child);

    std::vector<int> forwardCalculation();
};


#endif // COMPUTATIONAL_GRAPH_WITH_TENSOR_H
