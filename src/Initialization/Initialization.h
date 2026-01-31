//
// Created by Olcay YILDIZ on 27.01.2026.
//

#ifndef COMPUTATIONALGRAPH_INITIALIZATION_H
#define COMPUTATIONALGRAPH_INITIALIZATION_H
#include <vector>
#include <random>
using namespace std;

class Initialization {
public:
    virtual ~Initialization() = default;
    virtual vector<double> initialize(int row, int column, default_random_engine& randomEngine) = 0;
};

#endif //COMPUTATIONALGRAPH_INITIALIZATION_H