#ifndef COMPUTATIONAL_GRAPH_HPP
#define COMPUTATIONAL_GRAPH_HPP

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>

#include "ComputationalNode.h"
#include "FunctionType.h"
#include "Matrix.h"

class ComputationalGraph {
private:
    std::unordered_map<ComputationalNode*, std::vector<ComputationalNode*>> nodeMap;
    std::unordered_map<ComputationalNode*, std::vector<ComputationalNode*>> reverseNodeMap;

    void sort(ComputationalNode* node, std::unordered_set<ComputationalNode*>& visited,
              std::list<ComputationalNode*>& sortedNodes);
    void update(ComputationalNode* node, std::unordered_set<ComputationalNode*>& visited);

public:
    ComputationalGraph();

    ComputationalNode* addEdge(ComputationalNode* first, ComputationalNode* second, bool isBiased);
    ComputationalNode* addEdge(ComputationalNode* node, FunctionType type, bool isBiased);

    std::list<ComputationalNode*> topologicalSort();
    void updateValues();

    Matrix* calculateDerivative(ComputationalNode* node, ComputationalNode* child);
    std::vector<int> forwardCalculation();
};

#endif // COMPUTATIONAL_GRAPH_HPP
