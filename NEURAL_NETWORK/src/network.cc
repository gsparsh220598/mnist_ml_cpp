#include "network.hpp"
#include "layer.hpp"
#include "data_handler.hpp"
#include <numeric>

Network::Network(std::vector<int> spec, int input_size, int num_cls, double lr)
{
    for (int i = 0; i < spec.size(); ++i)
    {
        if (i == 0)
        {
            layers.push_back(new Layer(input_size, spec[i]));
        }
        else
        {
            layers.push_back(new Layer(layers[i - 1]->neurons.size(), spec[i]));
        }
    }
    layers.push_back(new Layer(layers[layers.size() - 1]->neurons.size(), num_cls));
    this->lr = lr;
}

Network::~Network() {}

double Network::activate(std::vector<double> weights, std::vector<double> input)
{
    double activation = weights.back(); // bias
    for (int i = 0; i < weights.size() - 1; ++i)
    {
        activation += weights[i] * input[i]; // dot product
    }
    return activation;
}

double Network::transfer(double activation)
{
    return 1.0 / (1.0 + exp(-activation)); // sigmoid
}

double Network::transfer_derivative(double output)
{
    return output * (1.0 - output); // derivative of sigmoid, used for back propagation
}

std::vector<double> Network::feed_forward(data *data)
{
    std::vector<double> inputs = *data->get_normalized_feature_vector();
    for (int i = 0; i < layers.size(); ++i) // for each layer
    {
        Layer *layer = layers[i];
        std::vector<double> new_inputs;
        for (Neuron *neuron : layer->neurons) // for each neuron in layer
        {
            double activation = this->activate(neuron->weights, inputs);
            neuron->output = this->transfer(activation);
            new_inputs.push_back(neuron->output); // output of neuron is input to next layer
        }
        inputs = new_inputs; // output of layer is input to next layer
    }
    return inputs; // output of last layer, prediction
}

void Network::back_propagate(data *data)
{
    for (int i = layers.size() - 1; i >= 0; --i)
    {
        Layer *layer = layers[i];
        std::vector<double> errors;
        if (i != layers.size() - 1) // not last layer
        {
            // calculate the contibution to the error from each neuron in the next layer
            for (int j = 0; j < layer->neurons.size(); ++j)
            {
                double error = 0.0;
                for (Neuron *neuron : layers[i + 1]->neurons)
                {
                    error += (neuron->weights[j] * neuron->delta);
                }
                errors.push_back(error);
            }
        }
        else // last layer
        {
            // calculate the error for each neuron in the last layer
            for (int j = 0; j < layer->neurons.size(); ++j)
            {
                Neuron *neuron = layer->neurons[j];
                errors.push_back(data->get_class_vector().at(j) - neuron->output);
            }
        }
        // calculate the delta (gradient) for each neuron in this layer
        for (int j = 0; j < layer->neurons.size(); ++j)
        {
            Neuron *neuron = layer->neurons[j];
            neuron->delta = errors[j] * this->transfer_derivative(neuron->output);
        }
    }
}

void Network::update_weights(data *data)
{
    std::vector<double> inputs = *data->get_normalized_feature_vector();
    for (int i = 0; i < layers.size(); ++i)
    {
        if (i != 0)
        {
            for (Neuron *neuron : layers[i - 1]->neurons)
            {
                inputs.push_back(neuron->output);
            }
        }
        for (Neuron *neuron : layers[i]->neurons)
        {
            for (int j = 0; j < inputs.size(); ++j)
            {
                neuron->weights[j] += this->lr * neuron->delta * inputs[j];
            }
            neuron->weights.back() += this->lr * neuron->delta; // bias
        }
        inputs.clear();
    }
}

int Network::predict(data *data)
{
    std::vector<double> outputs = this->feed_forward(data);
    // returns the position of the max value in the output vector,
    // which corresponds to the index of the class with the highest probability
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

void Network::train(int epochs)
{
    for (int i = 0; i < epochs; ++i) // for each epoch
    {
        double sum_error = 0.0;              // sum of errors for each epoch
        for (data *d : *this->training_data) // for each data point
        {
            std::vector<double> outputs = feed_forward(d);
            std::vector<int> expected = d->get_class_vector();
            double temp_err = 0.0;
            for (int i = 0; i < outputs.size(); ++i)
            {
                temp_err += pow(expected[i] - outputs[i], 2); // sum of squared errors
            }
            sum_error += temp_err;
            this->back_propagate(d);
            this->update_weights(d);
        }
        printf("epoch: %d error: %.4f\n", i, sum_error);
    }
}

double Network::test()
{
    double num_correct = 0.0;
    double count = 0.0;
    for (data *ts : *this->testing_data)
    {
        count++;
        int index = this->predict(ts);
        if (ts->get_class_vector().at(index) == 1) // if the prediction is correct
            num_correct++;
    }
    test_perf = num_correct / count;
    return test_perf;
}

void Network::validate()
{
    double num_correct = 0.0;
    double count = 0.0;
    for (data *vs : *this->validation_data)
    {
        count++;
        int index = this->predict(vs);
        if (vs->get_class_vector().at(index) == 1) // if the prediction is correct
            num_correct++;
    }
    double val_perf = num_correct / count;
    printf("Validation performance: %.4f\n", val_perf);
}

int main()
{
    data_handler *dh = new data_handler();
#ifdef MNIST
    dh->read_feature_vector("../train-images-idx3-ubyte");
    dh->read_feature_labels("../train-labels-idx1-ubyte");
    dh->count_classes();
#else
    dh->read_csv("../iris.data", ",");
#endif
    dh->split_data();
    std::vector<int> hidden_layers = {10};
    auto lambda = [&]()
    {
        Network *nn = new Network(
            hidden_layers,
            dh->get_training_data()->at(0)->get_normalized_feature_vector()->size(),
            dh->get_class_counts(),
            0.25);
        nn->set_training_data(dh->get_training_data());
        nn->set_test_data(dh->get_testing_data());
        nn->set_validation_data(dh->get_validation_data());
        nn->train(10);
        nn->validate();
        printf("Test performance: %.4f\n", nn->test());
    };
    lambda();
}