#include "../include/knn.hpp"
#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "data_handler.hpp"

knn::knn(int val)
{
    k = val;
}

knn::knn()
{
    // Default constructor, No need to initialize anything
}

knn::~knn()
{
    // No need to delete anything
}

// Complexity Analysis: O(N^2) if K ~ N
//  if K << N, then O(N) (eg. K=2)
//  O(NlogN) if we sort the distances
void knn::find_knearest(data *query_point)
{
    neighbors = new std::vector<data *>;
    double min = std::numeric_limits<double>::max();
    double prev_min = min;
    int idx = 0;
    for (int i = 0; i < k; i++)
    {
        if (i == 0)
        {
            for (int j = 0; j < train_set->size(); j++)
            {
                double distance = calc_distance(query_point, train_set->at(j)); // Calculate the distance between the query point and the data point
                train_set->at(j)->set_distance(distance);                       // Set the distance of the data point
                if (distance < min)                                             // Find the nearest neighbor
                {
                    min = distance;
                    idx = j;
                }
            }
            neighbors->push_back(train_set->at(idx)); // Add the nearest neighbor to the vector
            prev_min = min;                           // Save the previous min
            min = std::numeric_limits<double>::max(); // Reset min
        }
        else
        {
            for (int j = 0; j < train_set->size(); j++)
            {

                double distance = train_set->at(j)->get_distance(); // Calculate the distance between the query point and the data point
                train_set->at(j)->set_distance(distance);           // Set the distance of the data point
                if (distance < min && distance > prev_min)          // Find the nearest neighbor
                {
                    min = distance;
                    idx = j;
                }
            }
            neighbors->push_back(train_set->at(idx)); // Add the nearest neighbor to the vector
            prev_min = min;                           // Save the previous min
            min = std::numeric_limits<double>::max(); // Reset min
        }
    }
}

void knn::set_k(int val)
{
    k = val;
}

int knn::predict()
{
    std::map<uint8_t, int> cls_freq;
    for (int i = 0; i < neighbors->size(); i++) // Count the frequency of each class
    {
        if (cls_freq.find(neighbors->at(i)->get_label()) == cls_freq.end())
        {
            cls_freq[neighbors->at(i)->get_label()] = 1;
        }
        else
        {
            cls_freq[neighbors->at(i)->get_label()]++;
        }
    }

    int best = 0;
    int max = 0;
    for (auto kv : cls_freq) // Find the class with the most frequency
    {
        // printf("Class: %d, Frequency: %d\n", kv.first, kv.second);
        if (kv.second > max)
        {
            max = kv.second;
            best = kv.first;
        }
    }
    neighbors->clear();
    return best;
}

double knn::calc_distance(data *query_point, data *input)
{
    double distance = 0.0;
    if (query_point->get_feature_vector_size() != input->get_feature_vector_size())
    {
        printf("Error: Feature vector sizes do not match\n");
        exit(1);
    }
#ifdef EUCLID
    for (unsigned i = 0; i < query_point->get_feature_vector_size(); i++)
    {
        distance += pow(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i), 2);
    }
    distance = sqrt(distance);
    // printf("Distance: %f\n", distance);
#elif defined MANHATTAN
    for (unsigned i = 0; i < query_point->get_feature_vector_size(); i++)
    {
        distance += abs(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i));
    }
#endif
    return distance;
}

double knn::validate_performance()
{
    double curr_perf = 0;
    int count = 0;
    int data_idx = 0;
    for (data *query_point : *val_set) // Iterate through the validation set
    {
        find_knearest(query_point);
        int prediction = predict();
        // printf("Prediction: %d, Actual: %d\n", prediction, query_point->get_label());
        if (prediction == query_point->get_label())
        {
            count++;
        }
        data_idx++;
        printf("Current Performance: %.3f %%\n", ((double)count * 100) / ((double)data_idx));
    }
    curr_perf = ((double)count * 100) / ((double)val_set->size());
    printf("Valiation Performance for K = %d: %.3f %%\n", k, curr_perf);
    return curr_perf;
}

double knn::test_performance()
{
    double curr_perf = 0;
    int count = 0;
    int data_idx = 0;
    for (data *query_point : *test_set) // Iterate through the test set
    {
        find_knearest(query_point);
        int prediction = predict();
        if (prediction == query_point->get_label())
        {
            count++; // Count the number of correct predictions
        }
        data_idx++; // Count the number of data points
        printf("Current Performance: %.3f %%\n", ((double)count * 100) / ((double)data_idx));
    }
    curr_perf = ((double)count * 100) / ((double)test_set->size());
    printf("Testing Performance: %.3f %%\n", curr_perf);
    return curr_perf;
}

int main()
{
    data_handler *dh = new data_handler();
    dh->read_feature_vector("../train-images.idx3-ubyte");
    dh->read_feature_labels("../train-labels.idx1-ubyte");
    dh->split_data();
    dh->count_classes();
    knn *knn_classifier = new knn();
    knn_classifier->set_training_set(dh->get_training_data());
    knn_classifier->set_test_set(dh->get_testing_data());
    knn_classifier->set_validation_set(dh->get_validation_data());
    double perf = 0.0;
    double best_perf = 0.0;
    int best_k = 0;
    for (int i = 1; i <= 4; i++)
    {
        if (i == 1)
        {
            knn_classifier->set_k(i);
            perf = knn_classifier->validate_performance();
            best_perf = perf;
        }
        else
        {
            knn_classifier->set_k(i);
            perf = knn_classifier->validate_performance();
            if (perf > best_perf)
            {
                best_perf = perf;
                best_k = i;
            }
        }
        knn_classifier->set_k(best_k);
        knn_classifier->test_performance();
    }
}