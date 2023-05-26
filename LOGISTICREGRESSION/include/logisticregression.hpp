#ifndef __LOGISTICREGRESSION_HPP
#define __LOGISTICREGRESSION_HPP

#include "common.hpp"
#include <unordered_set>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <map>
#include "data_handler.hpp"

class logisticregression : public common_data
{
private:
    std::vector<double> weights;
    double learningRate;
    int numIterations;

public:
    logisticregression(double lr, int numIter);
    ~logisticregression();

    void train(const std::vector<data *> *features, const std::vector<int> *labels);
    double predict(const std::vector<data *> *features);

private:
    void initializeWeights(int numFeatures);
    double sigmoid(double z);
    // double dotProduct(const std::vector<data *> *features);
    // double calcAccuracy(const std::vector<data *> *features, const std::vector<int> *labels);
};

#endif // __LOGISTICREGRESSION_HPP