#ifndef __COMMON_HPP
#define __COMMON_HPP

#include <vector>
#include "data.hpp"

class common_data
{
protected:
    std::vector<data *> *training_data;
    std::vector<data *> *testing_data;
    std::vector<data *> *validation_data;

public:
    void set_training_data(std::vector<data *> *vect);
    void set_test_data(std::vector<data *> *vect);
    void set_validation_data(std::vector<data *> *vect);
};

#endif // __COMMON_HPP