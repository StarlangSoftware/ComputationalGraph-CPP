#ifndef SOFTMAX_H
#define SOFTMAX_H
#include <Matrix.h>

#include "Function.h"

class Softmax : public Function {
public:
    Matrix calculate(const Matrix& matrix) const override;
    Matrix derivative(const Matrix& matrix) const override;
};
#endif // SOFTMAX_H
