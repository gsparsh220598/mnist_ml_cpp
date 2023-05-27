#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include "data.hpp"
#include "neuron.hpp"
#include "layer.hpp"
#include "common.hpp"

class Network : public common_data
{
public:
    std::vector<Layer *> layers;
    double lr;
    double test_perf;
    Network(std::vector<int> spec, int, int, double);
    ~Network();
    std::vector<double> feed_forward(data *data);
    double activate(std::vector<double>, std::vector<double>); // dot product
    double transfer(double);                                   // sigmoid
    double transfer_derivative(double);                        // derivative of sigmoid, used for back propagation
    void back_propagate(data *data);
    void update_weights(data *data);
    int predict(data *data); // returns index of max value
    void train(int);         // number of epochs
    double test();
    void validate();
};

#endif