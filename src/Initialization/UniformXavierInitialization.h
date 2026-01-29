//
// Created by Olcay YILDIZ on 27.01.2026.
//

#ifndef COMPUTATIONALGRAPH_UNIFORMXAVIERINITIALIZATION_H
#define COMPUTATIONALGRAPH_UNIFORMXAVIERINITIALIZATION_H
#include "Initialization.h"


class UniformXavierInitialization : public Initialization {
public:
    vector<double> initialize(int row, int column, default_random_engine &randomEngine);
};


#endif //COMPUTATIONALGRAPH_UNIFORMXAVIERINITIALIZATION_H