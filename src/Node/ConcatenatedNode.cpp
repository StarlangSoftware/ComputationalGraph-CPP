//
// Created by Olcay YILDIZ on 27.01.2026.
//

#include "ConcatenatedNode.h"

ConcatenatedNode::ConcatenatedNode(int dimension)
    : ComputationalNode(false, false, nullptr, Tensor({0})) {
  this->nodeType = CONCATENATED_NODE_TYPE;
  this->dimension = dimension;
  this->valueNull = true;
}

int ConcatenatedNode::getDimension() const { return this->dimension; }

int ConcatenatedNode::getIndex(const ComputationalNode &node) const {
  return this->indexMap.at(&node);
}

void ConcatenatedNode::addNode(const ComputationalNode *node) {
  this->indexMap[node] = this->indexMap.size();
}
