//
// Created by Olcay YILDIZ on 27.01.2026.
//

#ifndef COMPUTATIONALGRAPH_HEUNIFORMINITIALIZATION_H
#define COMPUTATIONALGRAPH_HEUNIFORMINITIALIZATION_H
#include "Initialization.h"


class HeUniformInitialization : public Initialization{
public:
    vector<double> initialize(int row, int column, default_random_engine& randomEngine);
};


#endif //COMPUTATIONALGRAPH_HEUNIFORMINITIALIZATION_H