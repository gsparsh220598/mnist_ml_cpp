#ifndef __COMMON_HPP
#define __COMMON_HPP

#include <vector>
#include "data.hpp"

class common_data
{
protected:
    std::vector<data *> *train_set;
    std::vector<data *> *test_set;
    std::vector<data *> *val_set;

public:
    void set_training_set(std::vector<data *> *vect);
    void set_test_set(std::vector<data *> *vect);
    void set_validation_set(std::vector<data *> *vect);
};

#endif // __COMMON_HPP