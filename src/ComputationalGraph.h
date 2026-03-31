//
// Created by Olcay Taner YILDIZ on 10.04.2025.
//

#ifndef COMPUTATIONAL_GRAPH_WITH_TENSOR_H
#define COMPUTATIONAL_GRAPH_WITH_TENSOR_H

#include <map>
#include <queue>
#include <vector>

#include "NeuralNetworkParameter.h"
#include "Node/ComputationalNode.h"
#include "Node/MultiplicationNode.h"
#include "Performance/ClassificationPerformance.h"
#include "Tensor.h"

using namespace std;

class ComputationalGraph {
public:
  map<ComputationalNode *, vector<ComputationalNode *>> nodeMap;
  map<ComputationalNode *, vector<ComputationalNode *>> reverseNodeMap;
  void
  computeIfAbsent(map<ComputationalNode *, vector<ComputationalNode *>> &map,
                  ComputationalNode *first, ComputationalNode *second);

protected:
  deque<ComputationalNode *> sortRecursive(ComputationalNode *node,
                                           set<ComputationalNode *> &visited);
  deque<ComputationalNode *> topologicalSort();
  void clearRecursive(set<ComputationalNode *> &visited,
                      ComputationalNode *node);
  void clear();
  vector<int> transposeAxes(int length);
  Tensor getBiasedPartial(const Tensor &tensor);
  Tensor calculateDerivative(ComputationalNode *node, ComputationalNode *child);
  void calculateRMinusY(ComputationalNode *output,
                        const vector<int> &classLabelIndex);
  void getBiased(ComputationalNode *tensor);

public:
  /**
   * Trains the computational graph using the given training set and parameters.
   * @param trainSet The training set.
   * @param parameters The parameters of the computational graph.
   */
  virtual void train(vector<Tensor> &trainSet,
                     NeuralNetworkParameter &parameters) = 0;

  /**
   * Tests the computational graph on the given test set.
   * @param testSet The test set.
   * @return The classification performance of the computational graph on the
   * test set.
   */
  virtual ClassificationPerformance test(const vector<Tensor> &testSet) = 0;
  ComputationalNode *addMultiplicationEdge(ComputationalNode *first,
                                           MultiplicationNode *second,
                                           bool isBiased);
  ComputationalNode *addFunctionEdge(ComputationalNode *first,
                                     Function *function, bool isBiased = false);
  ComputationalNode *addNodeEdge(ComputationalNode *first,
                                 ComputationalNode *second, bool isBiased,
                                 bool isHadamard);
  ComputationalNode *addAdditionEdge(ComputationalNode *first,
                                     ComputationalNode *second, bool isBiased);
  ComputationalNode *concatEdges(const vector<ComputationalNode *> &nodes,
                                 int dimension);
  ~ComputationalGraph();

protected:
  vector<ComputationalNode *> inputNodes;
  ComputationalNode *outputNode = nullptr;
  /**
   * Retrieves the class label indexes associated with the given output node in
   * the computational graph.
   * @param outputNode The output node for which the class label indexes are to
   * be retrieved.
   * @return A list of integers representing the class label indexes.
   */
  vector<int> forwardCalculation(bool enableDropout);
  virtual vector<int> getClassLabels(ComputationalNode *outputNode) = 0;
  void backpropagation(Optimizer *optimizer,
                       const vector<int> &classLabelIndex);
  [[nodiscard]] vector<int> predict();
  vector<int> forwardCalculation();
};

#endif // COMPUTATIONAL_GRAPH_WITH_TENSOR_H
