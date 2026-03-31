#include "ComputationalGraph.h"
#include "Node/ConcatenatedNode.h"
#include <map>
#include <utility>

void ComputationalGraph::computeIfAbsent(
    map<ComputationalNode *, vector<ComputationalNode *>> &map,
    ComputationalNode *first, ComputationalNode *second) {
  vector<ComputationalNode *> newNodes;
  if (map.contains(first)) {
    newNodes = map[first];
  }
  newNodes.push_back(second);
  map[first] = newNodes;
}

/**
 * Recursive helper function to perform depth-first search for topological
 * sorting.
 * @param node The current node being processed.
 * @param visited A set of visited nodes.
 * @return A list representing the partial topological order.
 */
deque<ComputationalNode *>
ComputationalGraph::sortRecursive(ComputationalNode *node,
                                  set<ComputationalNode *> &visited) {
  deque<ComputationalNode *> queue;
  visited.emplace(node);
  if (nodeMap.contains(node)) {
    for (const auto child : nodeMap[node]) {
      if (!visited.contains(child)) {
        deque<ComputationalNode *> result = sortRecursive(child, visited);
        while (!result.empty()) {
          queue.emplace_back(result.front());
          result.pop_front();
        }
      }
    }
  }
  queue.emplace_back(node);
  return queue;
}

/**
 * Performs topological sorting on the computational graph.
 * @return A list representing the topological order of the nodes.
 */
deque<ComputationalNode *> ComputationalGraph::topologicalSort() {
  deque<ComputationalNode *> sortedList;
  set<ComputationalNode *> visited;
  for (const auto &node : nodeMap) {
    if (!visited.contains(node.first)) {
      deque<ComputationalNode *> queue = sortRecursive(node.first, visited);
      while (!queue.empty()) {
        sortedList.emplace_back(queue.front());
        queue.pop_front();
      }
    }
  }
  return sortedList;
}

/**
 * Recursive helper function to clear the values and gradients of nodes.
 */
void ComputationalGraph::clearRecursive(set<ComputationalNode *> &visited,
                                        ComputationalNode *node) {
  visited.emplace(node);
  if (!node->isLearnable()) {
    node->setValueNull();
  }
  node->setBackwardNull();
  if (nodeMap.contains(node)) {
    for (const auto child : nodeMap[node]) {
      if (!visited.contains(child)) {
        clearRecursive(visited, child);
      }
    }
  }
}

/**
 * Clears the values and gradients of all nodes in the graph.
 */
void ComputationalGraph::clear() {
  set<ComputationalNode *> visited;
  for (const auto &node : nodeMap) {
    if (!visited.contains(node.first)) {
      clearRecursive(visited, node.first);
    }
  }
}

vector<int> ComputationalGraph::transposeAxes(const int length) {
  vector<int> axes;
  axes.reserve(length);
  for (int i = 0; i < length - 2; i++) {
    axes.emplace_back(i);
  }
  axes.emplace_back(length - 1);
  axes.emplace_back(length - 2);
  return axes;
}

/**
 * Removes the bias term from the tensor.
 * @param tensor for which the bias term needs to be removed.
 * @return Tensor without bias term.
 */
Tensor ComputationalGraph::getBiasedPartial(const Tensor &tensor) {
  vector<int> endIndexes;
  vector<int> startIndexes;
  endIndexes.reserve(tensor.getShape().size());
  startIndexes.reserve(tensor.getShape().size());
  for (int i = 0; i < tensor.getShape().size(); i++) {
    startIndexes.emplace_back(0);
    if (i == tensor.getShape().size() - 1) {
      endIndexes.emplace_back(tensor.getShape()[i] - 1);
    } else {
      endIndexes.emplace_back(tensor.getShape()[i]);
    }
  }
  return tensor.partial(startIndexes, endIndexes);
}

/**
 * Calculates the derivative of the child node with respect to the parent node.
 * @param node Parent node.
 * @param child Child node.
 * @return The gradient tensor.
 */
Tensor ComputationalGraph::calculateDerivative(ComputationalNode *node,
                                               ComputationalNode *child) {
  vector<ComputationalNode *> reverseChildren = reverseNodeMap[child];
  auto backward = Tensor({0});
  if (child->isBiased()) {
    backward = getBiasedPartial(child->getBackward());
  } else {
    backward = child->getBackward();
  }
  if (child->getFunction() != nullptr) {
    Function *function = child->getFunction();
    Tensor childValue = Tensor({0});
    if (child->isBiased()) {
      childValue = getBiasedPartial(child->getValue());
    } else {
      childValue = child->getValue();
    }
    return function->derivative(childValue, backward);
  } else {
    if (child->getNodeType() == CONCATENATED_NODE_TYPE) {
      auto *_child = (ConcatenatedNode *)child;
      int index = _child->getIndex(*node);
      int blockSize =
          backward.getShape()[_child->getDimension()] / reverseChildren.size();
      int dimensions = blockSize;
      vector<int> shape;
      shape.reserve(backward.getShape().size());
      for (int i = 0; i < backward.getShape().size(); i++) {
        if (_child->getDimension() > i) {
          shape.emplace_back(backward.getShape()[i]);
        } else {
          if (_child->getDimension() < i) {
            dimensions *= backward.getShape()[i];
            shape.emplace_back(backward.getShape()[i]);
          } else {
            shape.emplace_back(blockSize);
          }
        }
      }
      vector<double> childValues = backward.getData();
      vector<double> newValues;
      int i = index * dimensions;
      while (i < childValues.size()) {
        for (int k = 0; k < dimensions; k++) {
          newValues.emplace_back(childValues[i + k]);
        }
        i += reverseChildren.size() * dimensions;
      }
      return Tensor(newValues, shape);
    } else {
      if (child->getNodeType() == MULTIPLICATION_NODE_TYPE) {
        auto *_child = (MultiplicationNode *)child;
        ComputationalNode *left = reverseChildren[0];
        ComputationalNode *right = reverseChildren[1];
        if (left == node) {
          Tensor rightValue = right->getValue();
          if (_child->isHadamard()) {
            return rightValue.hadamardProduct(backward);
          }
          return backward.multiply(rightValue.transpose(
              transposeAxes(rightValue.getShape().size())));
        }
        Tensor leftValue = left->getValue();
        if (_child->isHadamard()) {
          return leftValue.hadamardProduct(backward);
        }
        if (!left->isValueNull()) {
          return leftValue.transpose(transposeAxes(leftValue.getShape().size()))
              .multiply(backward);
        }
      }
      return backward;
    }
  }
}

ComputationalNode *ComputationalGraph::addMultiplicationEdge(
    ComputationalNode *first, MultiplicationNode *second, const bool isBiased) {
  ComputationalNode *newNode =
      new MultiplicationNode(false, isBiased, second->isHadamard(), first);
  computeIfAbsent(nodeMap, first, newNode);
  computeIfAbsent(reverseNodeMap, newNode, first);
  computeIfAbsent(nodeMap, second, newNode);
  computeIfAbsent(reverseNodeMap, newNode, second);
  return newNode;
}

ComputationalNode *ComputationalGraph::addFunctionEdge(ComputationalNode *first,
                                                       Function *function,
                                                       const bool isBiased) {
  return function->addToGraph({first}, isBiased, this);
}

ComputationalNode *ComputationalGraph::addNodeEdge(ComputationalNode *first,
                                                   ComputationalNode *second,
                                                   const bool isBiased,
                                                   bool isHadamard) {
  auto *newNode = new MultiplicationNode(false, isBiased, isHadamard, first);
  computeIfAbsent(nodeMap, first, newNode);
  computeIfAbsent(reverseNodeMap, newNode, first);
  computeIfAbsent(nodeMap, second, newNode);
  computeIfAbsent(reverseNodeMap, newNode, second);
  return newNode;
}

ComputationalNode *ComputationalGraph::addAdditionEdge(
    ComputationalNode *first, ComputationalNode *second, const bool isBiased) {
  auto *newNode = new ComputationalNode(false, nullptr, isBiased);
  computeIfAbsent(nodeMap, first, newNode);
  computeIfAbsent(reverseNodeMap, newNode, first);
  computeIfAbsent(nodeMap, second, newNode);
  computeIfAbsent(reverseNodeMap, newNode, second);
  return newNode;
}

/**
 * Concatenates the given nodes along the given dimension.
 * @param nodes List of nodes to be concatenated.
 * @param dimension Dimension along which the nodes need to be concatenated.
 * @return A new node that connects to the given nodes.
 */
ComputationalNode *
ComputationalGraph::concatEdges(const vector<ComputationalNode *> &nodes,
                                const int dimension) {
  auto *newNode = new ConcatenatedNode(dimension);
  for (const auto node : nodes) {
    computeIfAbsent(nodeMap, node, newNode);
    computeIfAbsent(reverseNodeMap, newNode, node);
    newNode->addNode(node);
  }
  return newNode;
}

void ComputationalGraph::backpropagation(Optimizer *optimizer,
                                         const vector<int> &classLabelIndex) {
  deque<ComputationalNode *> sortedNodes = topologicalSort();
  ComputationalNode *outputNode = sortedNodes.front();
  sortedNodes.pop_front();
  calculateRMinusY(outputNode, classLabelIndex);
  if (!sortedNodes.empty()) {
    ComputationalNode *first = sortedNodes.front();
    sortedNodes.pop_front();
    first->setBackward(outputNode->getBackward());
  }
  while (!sortedNodes.empty()) {
    ComputationalNode *node = sortedNodes.front();
    sortedNodes.pop_front();
    vector<ComputationalNode *> children = nodeMap[node];
    if (!children.empty()) {
      for (const auto child : children) {
        Tensor derivative = calculateDerivative(node, child);
        if (node->isBackwardNull()) {
          node->setBackward(derivative);
        } else {
          node->setBackward(node->getBackward().add(derivative));
        }
      }
    }
  }
  optimizer->updateValues(nodeMap);
  clear();
}

/**
 * Perform a forward pass and return predicted class indices.
 * @return A list of predicted class indices.
 */
vector<int> ComputationalGraph::predict() {
  vector<int> classLabels = forwardCalculation(false);
  clear();
  return classLabels;
}

/**
 * Perform a forward pass for the training phase.
 * @return A list of predicted class indices.
 */
vector<int> ComputationalGraph::forwardCalculation() {
  return forwardCalculation(true);
}

/**
 * Computes the difference between the predicted and actual values (R - Y).
 * @param output The output node of the computational graph.
 * @param classLabelIndex A list of true class labels (index of the correct
 * class for each sample).
 */
void ComputationalGraph::calculateRMinusY(ComputationalNode *output,
                                          const vector<int> &classLabelIndex) {
  vector<double> values;
  const vector<double> outputValues = output->getValue().getData();
  values.reserve(outputValues.size());
  int lastDimension =
      output->getValue().getShape()[output->getValue().getShape().size() - 1];
  for (int i = 0; i < outputValues.size(); i++) {
    if (i % lastDimension == classLabelIndex[i / lastDimension]) {
      values.push_back(1 - outputValues[i]);
    } else {
      values.push_back(-outputValues[i]);
    }
  }
  const Tensor backward = Tensor(values, output->getValue().getShape());
  output->setBackward(backward);
}

/**
 * Add a bias term to the node's value by appending a column of ones.
 * @param tensor The node whose value needs to be biased.
 */
void ComputationalGraph::getBiased(ComputationalNode *tensor) {
  int lastDimensionSize =
      tensor->getValue().getShape()[tensor->getValue().getShape().size() - 1];
  vector<double> values;
  vector<double> oldValues = tensor->getValue().getData();
  for (int i = 0; i < oldValues.size(); i++) {
    values.push_back(oldValues[i]);
    if ((i + 1) % lastDimensionSize == 0) {
      values.push_back(1.0);
    }
  }
  vector<int> shape;
  shape.reserve(tensor->getValue().getShape().size());
  for (int i = 0; i < tensor->getValue().getShape().size(); i++) {
    if (i == tensor->getValue().getShape().size() - 1) {
      shape.push_back(tensor->getValue().getShape()[i] + 1);
    } else {
      shape.push_back(tensor->getValue().getShape()[i]);
    }
  }
  auto biasedValue = Tensor(values, shape);
  tensor->setValue(biasedValue);
}

ComputationalGraph::~ComputationalGraph() {
  delete outputNode;
  for (const auto &node : nodeMap) {
    delete node.first;
  }
}

/**
 * Perform a forward pass through the computational graph.
 * @param enableDropout Whether to enable dropout or not.
 * @return A list of predicted class indices.
 */
vector<int> ComputationalGraph::forwardCalculation(bool enableDropout) {
  deque<ComputationalNode *> sortedNodes = topologicalSort();
  ComputationalNode *outputNode = sortedNodes.front();
  map<ComputationalNode *, vector<ComputationalNode *>> concatenatedNodeMap;
  map<ComputationalNode *, int> counterMap;
  while (sortedNodes.size() > 1) {
    ComputationalNode *currentNode = sortedNodes.back();
    sortedNodes.pop_back();
    if (currentNode->isBiased()) {
      getBiased(currentNode);
    }
    if (nodeMap.contains(currentNode)) {
      vector<ComputationalNode *> children = nodeMap[currentNode];
      for (const auto child : children) {
        if (child->isValueNull()) {
          if (child->getFunction() != nullptr) {
            Function *function = child->getFunction();
            Tensor currentValue = currentNode->getValue();
            if (function->type == FunctionType::DROPOUT) {
              if (enableDropout) {
                child->setValue(function->calculate(currentValue));
              } else {
                child->setValue(
                    Tensor(currentValue.getData(), currentValue.getShape()));
              }
            } else {
              child->setValue(function->calculate(currentValue));
            }
          } else {
            if (child->getNodeType() == CONCATENATED_NODE_TYPE) {
              if (!concatenatedNodeMap.contains(child)) {
                concatenatedNodeMap[child] =
                    vector<ComputationalNode *>(reverseNodeMap[child].size());
              }
              concatenatedNodeMap[child][((ConcatenatedNode *)child)
                                             ->getIndex(*currentNode)] =
                  currentNode;
              if (!counterMap.contains(child)) {
                counterMap[child] = 0;
              }
              counterMap[child]++;
              if (reverseNodeMap[child].size() == counterMap[child]) {
                child->setValue(concatenatedNodeMap[child][0]->getValue());
                for (size_t i = 1; i < concatenatedNodeMap[child].size(); i++) {
                  child->setValue(child->getValue().concat(
                      concatenatedNodeMap[child][i]->getValue(),
                      ((ConcatenatedNode *)child)->getDimension()));
                }
              }
            } else {
              child->setValue(currentNode->getValue());
            }
          }
        } else {
          if (child->getNodeType() == MULTIPLICATION_NODE_TYPE) {
            Tensor childValue = child->getValue();
            Tensor currentValue = currentNode->getValue();
            if (((MultiplicationNode *)child)->isHadamard()) {
              child->setValue(childValue.hadamardProduct(currentValue));
            } else {
              if (((MultiplicationNode *)child)->getPriorityNode() !=
                  currentNode) {
                child->setValue(childValue.multiply(currentValue));
              } else {
                child->setValue(currentValue.multiply(childValue));
              }
            }
          } else {
            Tensor result = child->getValue();
            Tensor currentValue = currentNode->getValue();
            child->setValue(result.add(currentValue));
          }
        }
      }
    }
  }
  return getClassLabels(outputNode);
}
