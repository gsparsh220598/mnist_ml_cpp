#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include "stdio.h"
#include "stdint.h"

class data
{
private:
    std::vector<uint8_t> *feature_vector;           // No class at end
    std::vector<double> *normalized_feature_vector; // No class at end
    std::vector<int> *class_vector;
    uint8_t label;
    uint8_t enum_label; // 0-9
    double distance;

public:
    data();
    ~data();
    void set_feature_vector(std::vector<uint8_t> *);
    void append_to_feature_vector(uint8_t);
    void append_to_feature_vector(double);
    void set_class_vector(int count);
    void set_normalized_feature_vector(std::vector<double> *);
    void set_label(uint8_t);
    void set_enum_label(uint8_t);
    void set_distance(double val);
    void print_vector();
    void print_normalized_vector();

    int get_feature_vector_size();
    uint8_t get_enum_label();
    double get_distance();
    uint8_t get_label();

    std::vector<uint8_t> *get_feature_vector();
    std::vector<double> *get_normalized_feature_vector();
    std::vector<int> get_class_vector();
};

#endif
