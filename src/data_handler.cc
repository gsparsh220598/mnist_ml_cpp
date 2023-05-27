#include "../include/data_handler.hpp"
#include <algorithm>
#include <random>

data_handler::data_handler()
{
    data_array = new std::vector<data *>();
    training_data = new std::vector<data *>();
    testing_data = new std::vector<data *>();
    validation_data = new std::vector<data *>();
}

data_handler::~data_handler()
{
    // Free dynamically allocated memory
}

void data_handler::read_csv(std::string path, std::string delimeter)
{
    num_classes = 0;
    std::ifstream data_file;
    // printf("Reading data from %s\n", path.c_str());
    data_file.open(path.c_str());
    printf("Reading data from %s\n", path.c_str());
    std::string line; // each line of the csv file

    while (std::getline(data_file, line))
    {
        if (line.length() == 0)
        {
            printf("Empty line\n");
            continue;
        }
        data *d = new data();
        d->set_normalized_feature_vector(new std::vector<double>());
        size_t position = 0;
        std::string token; // value in betwewen delimeter
        while ((position = line.find(delimeter)) != std::string::npos)
        {
            token = line.substr(0, position);
            d->append_to_feature_vector(std::stod(token)); // will throw exception if token is not a number, header
            line.erase(0, position + delimeter.length());
        }
        if (classMap.find(line) != classMap.end())
        {
            d->set_label(classMap[line]);
        }
        else
        {
            classMap[line] = num_classes;
            d->set_label(classMap[line]);
            num_classes++;
        }
        data_array->push_back(d);
    }
    for (data *d : *data_array)
        d->set_class_vector(num_classes);
    feature_vector_size = data_array->at(0)->get_normalized_feature_vector()->size();
}

void data_handler::read_feature_vector(std::string path)
{
    uint32_t header[4]; // 4 byte header (magic number, number of images, number of rows, number of columns)
    unsigned char bytes[4];
    printf("Reading feature vector from %s\n", path.c_str());
    FILE *file = fopen(path.c_str(), "rb"); // c_str() converts string to char array

    if (file == NULL)
    {
        printf("Error opening feature file\n");
        exit(1);
    }
    else
    {
        for (int i = 0; i < 4; i++)
        {
            if (fread(bytes, sizeof(bytes), 1, file))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done reading header\n");
        int image_size = header[2] * header[3];
        for (int i = 0; i < header[1]; i++)
        {
            data *d = new data();
            uint8_t element[1];
            for (int j = 0; j < image_size; j++)
            {
                if (fread(element, sizeof(element), 1, file))
                {
                    d->append_to_feature_vector(element[0]);
                }
                else
                {
                    printf("Error reading file\n");
                    exit(1);
                }
            }
            data_array->push_back(d); // data array stores pointer to data object
        }
        printf("Done reading %lu feature vector\n", data_array->size());
    }
}

void data_handler::read_feature_labels(std::string path)
{
    uint32_t header[2];     // 2 byte header (magic number, number of labels)
    unsigned char bytes[4]; // 4 bytes per header element
    FILE *file = fopen(path.c_str(), "rb");
    if (file == NULL)
    {
        printf("Error opening label file\n");
        exit(1);
    }
    else
    {
        for (int i = 0; i < 2; i++)
        {
            if (fread(bytes, sizeof(bytes), 1, file))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done reading header\n");
        for (int i = 0; i < header[1]; i++)
        {
            uint8_t element[1];
            if (fread(element, sizeof(element), 1, file))
            {
                data_array->at(i)->set_label(element[0]);
            }
            else
            {
                printf("Error reading file\n");
                exit(1);
            }
        }
        printf("Done reading labels\n");
    }
}

void data_handler::split_data()
{
    std::unordered_set<int> used_indices;
    int training_size = data_array->size() * TRAINING_DATA_SIZE;
    int testing_size = data_array->size() * TESTING_DATA_SIZE;
    int validation_size = data_array->size() * VALIDATION_DATA_SIZE;

    std::random_shuffle(data_array->begin(), data_array->end());

    // Training data
    int count = 0;
    int idx = 0;
    while (count < training_size)
    {
        training_data->push_back(data_array->at(idx));
        used_indices.insert(idx);
        count++;
        idx++;
    }

    // Testing data
    count = 0;
    while (count < testing_size)
    {
        int rand_index = rand() % data_array->size();
        if (used_indices.find(rand_index) == used_indices.end()) // If index not already used
        {
            testing_data->push_back(data_array->at(rand_index));
            used_indices.insert(rand_index);
            count++;
        }
    }

    // Validation data
    count = 0;
    while (count < validation_size)
    {
        int rand_index = rand() % data_array->size();
        if (used_indices.find(rand_index) == used_indices.end()) // If index not already used
        {
            validation_data->push_back(data_array->at(rand_index));
            used_indices.insert(rand_index);
            count++;
        }
    }

    printf("Training data size: %lu\n", training_data->size());
    printf("Testing data size: %lu\n", testing_data->size());
    printf("Validation data size: %lu\n", validation_data->size());
}

void data_handler::count_classes()
{
    int count = 0;
    for (unsigned i = 0; i < data_array->size(); i++)
    {
        if (class_map.find(data_array->at(i)->get_label()) == class_map.end())
        {
            class_map[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enum_label(count);
            count++;
        }
    }
    num_classes = count;
    for (data *data : *data_array)
    {
        data->set_class_vector(num_classes);
    }

    printf("Number of classes: %d\n", num_classes);
}

uint32_t data_handler::convert_to_little_endian(const unsigned char *bytes)
{
    // explain this
    // https://stackoverflow.com/questions/2182002/convert-big-endian-to-little-endian-in-c-without-using-provided-func
    return (uint32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
}

void data_handler::normalize()
{
    std::vector<double> mins, maxs;

    data *d = data_array->at(0);
    for (auto val : *d->get_feature_vector())
    {
        mins.push_back(val);
        maxs.push_back(val);
    }

    // Find min and max values for each feature
    for (int i = 1; i < data_array->size(); ++i)
    {
        d = data_array->at(i);
        for (int j = 0; j < d->get_feature_vector_size(); ++j) // For each feature
        {
            double value = (double)d->get_feature_vector()->at(j);
            if (value < mins[j])
                mins[j] = value;
            if (value > maxs[j])
                maxs[j] = value;
        }
    }

    // Normalize each feature
    for (int i = 0; i < data_array->size(); ++i)
    {
        data_array->at(i)->set_normalized_feature_vector(new std::vector<double>());
        data_array->at(i)->set_class_vector(num_classes);
        for (int j = 0; j < data_array->at(i)->get_feature_vector_size(); ++j)
        {
            if (maxs[j] - mins[j] == 0)
                data_array->at(i)->append_to_feature_vector(0.0);
            else
                data_array->at(i)->append_to_feature_vector(
                    (double)(data_array->at(i)->get_feature_vector()->at(j) - mins[j]) / (maxs[j] - mins[j]));
        }
    }
}

void data_handler::print_data()
{
    printf("Printing Training data\n");
    for (auto data : *training_data)
    {
        for (auto val : *data->get_normalized_feature_vector())
        {
            printf("%.3f ", val);
        }
        printf("-> %d\n", data->get_label());
    }
    return;

    printf("Printing Testing data\n");
    for (auto data : *testing_data)
    {
        for (auto val : *data->get_normalized_feature_vector())
        {
            printf("%.3f ", val);
        }
        printf("-> %d\n", data->get_label());
    }

    printf("Printing Validation data\n");
    for (auto data : *validation_data)
    {
        for (auto val : *data->get_normalized_feature_vector())
        {
            printf("%.3f ", val);
        }
        printf("-> %d\n", data->get_label());
    }
}

int data_handler::get_class_counts()
{
    return num_classes;
}

int data_handler::get_training_data_size()
{
    return training_data->size();
}

int data_handler::get_testing_data_size()
{
    return testing_data->size();
}

int data_handler::get_validation_data_size()
{
    return validation_data->size();
}

std::vector<data *> *data_handler::get_training_data()
{
    return training_data;
}

std::vector<data *> *data_handler::get_testing_data()
{
    return testing_data;
}

std::vector<data *> *data_handler::get_validation_data()
{
    return validation_data;
}

std::map<u_int8_t, int> data_handler::get_class_map()
{
    return class_map;
}