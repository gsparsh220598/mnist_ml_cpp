#ifndef __KMEANS_HPP
#define __KMEANS_HPP

#include "common.hpp"
#include <unordered_set>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <map>
#include "data_handler.hpp"

typedef struct cluster
{
    std::vector<double> *centroid;
    std::vector<data *> *cluster_points;
    std::map<int, int> class_counts;
    int most_freq_class;

    cluster(data *initial_point)
    {
        centroid = new std::vector<double>;
        cluster_points = new std::vector<data *>;
        for (auto value : *(initial_point->get_feature_vector()))
        {
            centroid->push_back(value);
        }
        cluster_points->push_back(initial_point);
        class_counts[initial_point->get_label()] = 1;
        most_freq_class = initial_point->get_label();
    }

    void add_to_cluster(data *point) // add a point to the cluster
    {
        int prev_size = cluster_points->size(); // for calculating new centroid
        cluster_points->push_back(point);       // add point to cluster
        for (int i = 0; i < centroid->size() - 1; i++)
        {
            double value = centroid->at(i);              // get previous value
            value *= prev_size;                          // multiply by previous size
            value += point->get_feature_vector()->at(i); // add new value
            value /= (double)cluster_points->size();     // divide by new size
            centroid->at(i) = value;                     // update centroid
        }
        if (class_counts.find(point->get_label()) == class_counts.end())
        {
            class_counts[point->get_label()] = 1;
        }
        else
        {
            class_counts[point->get_label()]++;
        }
        set_most_freq_cls();
    }

    void set_most_freq_cls()
    {
        int best_class;
        int freq = 0;
        for (auto kv : class_counts) // find most frequent class
        {
            if (kv.second > freq) // if current class is more frequent than previous
            {
                freq = kv.second;
                best_class = kv.first;
            }
        }
        most_freq_class = best_class;
    }

} cluster_t;

class kmeans : public common_data
{
    int num_clusters;
    std::vector<cluster_t *> *clusters;
    std::unordered_set<int> *used_idxs;

public:
    kmeans(int k);
    void init_clusters();
    void init_clusters_for_each_class();
    void train();
    double euclidean_distance(std::vector<double> *, data *);
    double validate();
    double test();
};

#endif // __KMEANS_HPP