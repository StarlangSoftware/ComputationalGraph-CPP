//
// Created by Olcay YILDIZ on 27.01.2026.
//

#ifndef COMPUTATIONALGRAPH_RANDOMINITIALIZATION_H
#define COMPUTATIONALGRAPH_RANDOMINITIALIZATION_H
#include "Initialization.h"


class RandomInitialization : public Initialization{
public:
    vector<double> initialize(int row, int column, default_random_engine &randomEngine);
};


#endif //COMPUTATIONALGRAPH_RANDOMINITIALIZATION_H