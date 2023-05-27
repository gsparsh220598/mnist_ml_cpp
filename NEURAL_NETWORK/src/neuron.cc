#include "neuron.hpp"
#include <random>

double generate_random_number(double min, double max)
{
    double random = ((double)rand()) / (double)RAND_MAX;
    return min + random * (max - min);
}

Neuron::Neuron(int prev_layer_size, int current_layer_size)
{
    init_weights(prev_layer_size);
}

void Neuron::init_weights(int prev_layer_size)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < prev_layer_size + 1; i++)
    {
        weights.push_back(generate_random_number(-1.0, 1.0));
    }
}