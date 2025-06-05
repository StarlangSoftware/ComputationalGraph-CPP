//
// Created by Yiğit Demirşan on 17.04.2025.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <cassert>
#include <Matrix.h>

Matrix createInputMatrix(const std::vector<std::string>& instance) {
    Matrix matrix(1, instance.size() - 1);
    for (size_t i = 0; i < instance.size() - 1; ++i) {
        matrix.setValue(0, i, std::stod(instance[i]));
    }
    return matrix;
}

std::vector<std::string> split(const std::string& line, char delimiter = ',') {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void test1() {
    std::ifstream file("iris.txt");
    if (!file.is_open()) {
        std::cerr << "Could not open iris.txt file!" << std::endl;
        exit(1);
    }

    std::map<std::string, int> labelMap;
    std::vector<std::vector<std::string>> dataSet, trainSet, testSet;
    std::string line;
    while (std::getline(file, line)) {
        auto row = split(line);
        dataSet.push_back(row);
        if (labelMap.find(row.back()) == labelMap.end()) {
            labelMap[row.back()] = labelMap.size();
        }
    }
    file.close();

    std::shuffle(dataSet.begin(), dataSet.end(), std::default_random_engine(1));
    for (size_t i = 0; i < dataSet.size(); ++i) {
        if (i < 120) trainSet.push_back(dataSet[i]);
        else testSet.push_back(dataSet[i]);
    }

    ComputationalGraph graph;
    ComputationalNode input(false, '*', true);

    Matrix m1(5, 4, -0.01, 0.01, std::default_random_engine(1));
    ComputationalNode w1(m1, '*');
    auto a1 = graph.addEdge(&input, &w1, true);
    auto a1Sigmoid = graph.addEdge(a1, FunctionType::SIGMOID, true);

    Matrix m2(5, 20, -0.01, 0.01, std::default_random_engine(1));
    ComputationalNode w2(m2, '*');
    auto a2 = graph.addEdge(a1Sigmoid, &w2, true);
    auto a2Sigmoid = graph.addEdge(a2, FunctionType::SIGMOID, true);

    Matrix m3(21, labelMap.size(), -0.01, 0.01, std::default_random_engine(1));
    ComputationalNode w3(m3, '*');
    auto a3 = graph.addEdge(a2Sigmoid, &w3, false);
    graph.addEdge(a3, FunctionType::SOFTMAX, false);

    int epochs = 1000;
    double learningRate = 0.1;
    double etaDecrease = 0.99;
    std::vector<int> classList;

    for (int i = 0; i < epochs; ++i) {
        std::shuffle(trainSet.begin(), trainSet.end(), std::default_random_engine(1));
        for (const auto& instance : trainSet) {
            input.setValue(createInputMatrix(instance));
            graph.forwardCalculation();
            classList.push_back(labelMap[instance.back()]);
            graph.backpropagation(learningRate, classList);
            classList.clear();
        }
        learningRate *= etaDecrease;
    }

    int correct = 0;
    for (const auto& test : testSet) {
        input.setValue(createInputMatrix(test));
        int prediction = graph.predict()[0];
        if (prediction == labelMap[test.back()]) {
            correct++;
        }
    }

    double accuracy = static_cast<double>(correct) / testSet.size();
    std::cout << "[Test1] Accuracy: " << accuracy << std::endl;
    assert(std::abs(accuracy - 0.966666) < 0.001);
}

void test2() {
    ComputationalGraph graph;
    ComputationalNode a0(false, '+', false);
    ComputationalNode a1(true, '+', false);
    auto a2 = graph.addEdge(&a0, &a1, false);
    auto output = graph.addEdge(a2, FunctionType::SOFTMAX, false);

    a0.setValue(Matrix(1, 3, 0, 100, std::default_random_engine(1)));
    a1.setValue(Matrix(1, 3, 0, 100, std::default_random_engine(1)));

    graph.forwardCalculation();
    std::vector<int> labels = {1};
    graph.backpropagation(0.01, labels);
    std::cout << "[Test2] Softmax forward & backprop ran without crash." << std::endl;
}

int main() {
    try {
        test1();
        test2();
        std::cout << "✅ All tests passed!" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "❌ Test failed: " << ex.what() << std::endl;
    } catch (...) {
        std::cerr << "❌ Test failed: Unknown error." << std::endl;
    }

    return 0;
}
