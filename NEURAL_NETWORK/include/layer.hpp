#ifndef __LAYER_HPP
#define __LAYER_HPP

#include <vector>
#include "neuron.hpp"
#include <stdint.h>

static int layerId = 0;

class Layer
{
public:
    int current_layer_size;
    std::vector<Neuron *> neurons;
    std::vector<double> layer_outputs;
    Layer(int, int);
    ~Layer();
    std::vector<double> get_layer_outputs();
    int get_size();
};

#endif