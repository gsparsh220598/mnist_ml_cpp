#ifndef __NEURON_HPP
#define __NEURON_HPP

#include <cmath>
#include <vector>
#include <stdio.h>

class Neuron
{

public:
    std::vector<double> weights;
    double output;
    double delta;
    Neuron(int, int);
    ~Neuron();
    void init_weights(int);
};

#endif