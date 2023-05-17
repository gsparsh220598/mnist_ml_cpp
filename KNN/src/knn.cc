#include "knn.hpp"
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
                double distance = calc_distance(query_point, train_set->at(j)); // Calculate the distance between the query point and the data point
                train_set->at(j)->set_distance(distance);                       // Set the distance of the data point
                if (distance < min && distance > prev_min)                      // Find the nearest neighbor
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

void knn::set_training_set(std::vector<data *> *vect)
{
    train_set = vect;
}

void knn::set_test_set(std::vector<data *> *vect)
{
    test_set = vect;
}

void knn::set_validation_set(std::vector<data *> *vect)
{
    val_set = vect;
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
        if (kv.second > max)
        {
            max = kv.second;
            best = kv.first;
        }
    }
    delete neighbors;
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
        distance = pow(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i), 2);
    }
    distance = sqrt(distance);

#elif defined MANHATTAN
    for (unsigned i = 0; i < query_point->get_feature_vector_size(); i++)
    {
        distance = abs(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i));
    }
#endif
    return distance;
}