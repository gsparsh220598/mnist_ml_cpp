#include "../include/logisticregression.hpp"
#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "data_handler.hpp"

logisticregression::logisticregression(double lr, int numIter)
{
    learningRate = lr;
    numIterations = numIter;
}

logisticregression::~logisticregression()
{
    // No need to delete anything
}

void logisticregression::train(const std::vector<data *> *features, const std::vector<int> *labels)
{
    // Initialize weights
    initializeWeights(features->at(0)->get_feature_vector_size());

    // Train the model
    for (int iter = 0; iter < numIterations; ++iter) // Iterate over the number of iterations
    {

        for (int i = 0; i < features->size(); ++i) // Iterate over all the data points
        {
            double yhat = predict(features->at(i)->get_feature_vector()); // Predict the label
            double error = labels->at(i) - yhat;                          // Calculate the error
            for (int k = 0; k < weights.size(); ++k)                      // Update all the weights
            {
                weights[k] += learningRate * error * features->at(j)->get_feature_vector()->at(k);
            }
        }
        if (i % 100 == 0)
        {
            std::cout << "Iteration: " << i << " Error: " << error << std::endl;
        }
    }
}

double logisticregression::predict(const std::vector<data *> *features)
{
    double z = 0.0;
    // Dot product
    for (int i = 0; i < features->size(); i++)
    {
        z += weights[i] * features->at(i)->get_feature_vector()->at(i);
    }
    return sigmoid(z);
}

void logisticregression::initializeWeights(int numFeatures)
{
    weights.resize(numFeatures, 0.0);
}

double logisticregression::sigmoid(double z)
{
    return 1.0 / (1.0 + exp(-z));
}

int main()
{
    // Read the data
    data_handler *dh = new data_handler();
    dh->read_data("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte", "../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte");

    // Get the training, test and validation sets
    std::vector<data *> *train_set = dh->get_training_set();
    std::vector<data *> *test_set = dh->get_test_set();
    std::vector<data *> *val_set = dh->get_validation_set();

    // Get the labels
    std::vector<int> *train_labels = dh->get_training_labels();
    std::vector<int> *test_labels = dh->get_test_labels();
    std::vector<int> *val_labels = dh->get_validation_labels();

    // Create the model
    logisticregression *lr = new logisticregression(0.01, 100);

    // Train the model
    lr->train(train_set, train_labels);

    // Test the model
    double accuracy = lr->predict(test_set);
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}