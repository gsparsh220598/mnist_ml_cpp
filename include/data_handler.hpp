#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include <fstream>
#include "stdio.h"
#include "data.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>

class data_handler
{
    std::vector<data *> *data_array;      // all of the data (pre-split)
    std::vector<data *> *training_data;   // training data
    std::vector<data *> *testing_data;    // testing data
    std::vector<data *> *validation_data; // validation data

    int num_classes;                  // number of classes
    int feature_vector_size;          // size of feature vector
    std::map<uint8_t, int> class_map; // maps class to number of instances
    std::map<std::string, int> classMap;

    const double TRAINING_DATA_SIZE = 0.75;
    const double TESTING_DATA_SIZE = 0.20;
    const double VALIDATION_DATA_SIZE = 0.05;

public:
    data_handler();
    ~data_handler();

    void read_csv(std::string path, std::string delimeter);
    void read_feature_vector(std::string path);
    void read_feature_labels(std::string path);
    void split_data();
    void count_classes();

    uint32_t convert_to_little_endian(const unsigned char *bytes);

    int get_class_counts();
    std::vector<data *> *get_training_data();
    std::vector<data *> *get_testing_data();
    std::vector<data *> *get_validation_data();
};

#endif