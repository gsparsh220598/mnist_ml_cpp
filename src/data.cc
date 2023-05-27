#include "../include/data.hpp"

data::data()
{
    feature_vector = new std::vector<uint8_t>();
}
data::~data()
{
    delete feature_vector;
}
void data::set_feature_vector(std::vector<uint8_t> *vect)
{
    feature_vector = vect;
}
void data::append_to_feature_vector(uint8_t val)
{
    feature_vector->push_back(val);
}
void data::set_normalized_feature_vector(std::vector<double> *vect)
{
    normalized_feature_vector = vect;
}
void data::append_to_feature_vector(double val)
{
    normalized_feature_vector->push_back(val);
}
void data::set_label(uint8_t val)
{
    label = val;
}

void data::print_vector()
{
    for (int i = 0; i < feature_vector->size(); i++)
    {
        printf("%d ", feature_vector->at(i));
    }
    printf("\n");
}

void data::print_normalized_vector()
{
    for (int i = 0; i < normalized_feature_vector->size(); i++)
    {
        printf("%f ", normalized_feature_vector->at(i));
    }
    printf("\n");
}

void data::set_enum_label(uint8_t val)
{
    enum_label = val;
}
void data::set_distance(double val)
{
    distance = val;
}
int data::get_feature_vector_size()
{
    return feature_vector->size();
}
uint8_t data::get_label()
{
    return label;
}

void data::set_class_vector(int count)
{
    class_vector = new std::vector<int>();
    for (int i = 0; i < count; i++)
    {
        if (i == label)
        {
            class_vector->push_back(1);
        }
        else
        {
            class_vector->push_back(0);
        }
    }
}

uint8_t data::get_enum_label()
{
    return enum_label;
}
std::vector<uint8_t> *data::get_feature_vector()
{
    return feature_vector;
}

std::vector<double> *data::get_normalized_feature_vector()
{
    return normalized_feature_vector;
}

std::vector<int> data::get_class_vector()
{
    return *class_vector;
}

double data::get_distance()
{
    return distance;
}
