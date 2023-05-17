#ifndef __KNN_H
#define __KNN_H

#include <vector>
#include "data.hpp"

class knn
{
    int k;
    std::vector<data *> *neighbors;
    std::vector<data *> *train_set;
    std::vector<data *> *test_set;
    std::vector<data *> *val_set;

public:
    knn(int);
    knn();
    ~knn();

    void find_knearest(data *query_point);
    void set_training_set(std::vector<data *> *vect);
    void set_test_set(std::vector<data *> *vect);
    void set_validation_set(std::vector<data *> *vect);

    int predict();
    double calc_distance(data *query_point, data *input);
    double validate_performance();
    double test_performance();
};

#endif // __KNN_H