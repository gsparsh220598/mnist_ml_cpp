#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include "stdio.h"
#include "stdint.h"

class data
{
    std::vector<uint8_t> *feature_vector; // No class at end
    uint8_t label;
    int enum_label; // 0-9
    double distance;

public:
    data();
    ~data();
    void set_feature_vector(std::vector<uint8_t> *);
    void append_to_feature_vector(uint8_t);
    void set_label(uint8_t);
    void set_enum_label(int);
    void set_distance(double);

    int get_feature_vector_size();
    uint8_t get_label();
    int get_enum_label();

    std::vector<uint8_t> *get_feature_vector();
};

#endif