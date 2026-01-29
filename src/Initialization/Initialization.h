//
// Created by Olcay YILDIZ on 27.01.2026.
//

#ifndef COMPUTATIONALGRAPH_INITIALIZATION_H
#define COMPUTATIONALGRAPH_INITIALIZATION_H
#include <vector>
#include <random>
using namespace std;

class Initialization {
    vector<double> initialize(int row, int column, const default_random_engine &randomEngine);
};

#endif //COMPUTATIONALGRAPH_INITIALIZATION_H