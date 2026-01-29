//
// Created by Olcay YILDIZ on 28.01.2026.
//

#ifndef COMPUTATIONALGRAPH_ADAMW_H
#define COMPUTATIONALGRAPH_ADAMW_H
#include "Adam.h"

class AdamW : public Adam{
private:
    double weightDecay;
public:
    AdamW(double learningRate, double etaDecrease, double beta1, double beta2, double epsilon, double weightDecay);
protected:
    void setGradients(ComputationalNode &node) override;
};


#endif //COMPUTATIONALGRAPH_ADAMW_H
