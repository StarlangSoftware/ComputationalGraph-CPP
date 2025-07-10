#ifndef COMPUTATIONAL_GRAPH_H
#define COMPUTATIONAL_GRAPH_H

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>

#include "Tensor.h"
#include "ComputationalNode.h"
#include "FunctionType.h"

class ComputationalGraph {
private:
    std::unordered_map<ComputationalNode*, std::vector<ComputationalNode*> > nodeMap;
    std::unordered_map<ComputationalNode*, std::vector<ComputationalNode*> > reverseNodeMap;

    void sort(ComputationalNode* node, std::unordered_set<ComputationalNode*>& visited, std::list<ComputationalNode*>& sortedNodes);
    void updateRecursive(ComputationalNode* node, std::unordered_set<ComputationalNode*>& visited);
    void clearRecursive(ComputationalNode* node, std::unordered_set<ComputationalNode*>& visited);

public:
    ComputationalGraph();

    ComputationalNode* addEdge(ComputationalNode* first, ComputationalNode* second, bool isBiased = false);
    ComputationalNode* addEdge(ComputationalNode* node, FunctionType type, bool isBiased = false);

    std::list<ComputationalNode*> topologicalSort();

    void updateValues();
    void clear();

    Tensor calculateDerivative(ComputationalNode* node, ComputationalNode* child);

    void getBiased(ComputationalNode* node);

    std::vector<int> forwardCalculation();

    std::vector<int> predict();

    // Additional methods for backpropagation/learning can be added as needed.
};

#endif // COMPUTATIONAL_GRAPH_H
