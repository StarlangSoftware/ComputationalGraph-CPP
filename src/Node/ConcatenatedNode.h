//
// Created by Olcay YILDIZ on 27.01.2026.
//

#ifndef COMPUTATIONALGRAPH_CONCATENATEDNODE_H
#define COMPUTATIONALGRAPH_CONCATENATEDNODE_H
#include "ComputationalNode.h"
#include <map>

class ConcatenatedNode : public ComputationalNode {
private:
  map<const ComputationalNode *, int> indexMap;
  int dimension;

public:
  ConcatenatedNode(int dimension);
  [[nodiscard]] int getDimension() const;
  [[nodiscard]] int getIndex(const ComputationalNode &node) const;
  void addNode(const ComputationalNode *node);
};

#endif // COMPUTATIONALGRAPH_CONCATENATEDNODE_H