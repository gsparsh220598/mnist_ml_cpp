#include "common.hpp"

void common_data::set_training_set(std::vector<data *> *vect)
{
    train_set = vect;
}

void common_data::set_test_set(std::vector<data *> *vect)
{
    test_set = vect;
}

void common_data::set_validation_set(std::vector<data *> *vect)
{
    val_set = vect;
}
