#include "../include/kmeans.hpp"

kmeans::kmeans(int k)
{
    num_clusters = k;
    clusters = new std::vector<cluster_t *>;
    used_idxs = new std::unordered_set<int>;
}

void kmeans::init_clusters()
{
    for (int i = 0; i < num_clusters; i++)
    {
        int idx = rand() % train_set->size();
        while (used_idxs->find(idx) != used_idxs->end()) // while idx is already used
        {
            idx = rand() % train_set->size();
        }
        clusters->push_back(new cluster_t(train_set->at(idx))); // add new cluster
        used_idxs->insert(idx);                                 // add idx to used_idxs
    }
}

void kmeans::init_clusters_for_each_class()
{
    std::unordered_set<int> used_classes;
    for (int i = 0; i < train_set->size(); i++)
    {
        if (used_classes.find(train_set->at(i)->get_label()) == used_classes.end()) // if class is not used
        {
            clusters->push_back(new cluster_t(train_set->at(i)));
            used_classes.insert(train_set->at(i)->get_label());
            used_idxs->insert(i);
        }
    }
}

void kmeans::train()
{
    while (used_idxs->size() < train_set->size())
    {
        int idx = rand() % train_set->size();
        while (used_idxs->find(idx) != used_idxs->end()) // while idx is already used
        {
            idx = rand() % train_set->size();
        }
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int i = 0; i < clusters->size(); i++)
        {
            double dist = euclidean_distance(clusters->at(i)->centroid, train_set->at(idx));
            if (dist < min_dist)
            {
                min_dist = dist;
                best_cluster = i;
            }
        }
        clusters->at(best_cluster)->add_to_cluster(train_set->at(idx)); // add point to cluster
        used_idxs->insert(idx);
    }
}

double kmeans::euclidean_distance(std::vector<double> *centroid, data *point)
{
    double dist = 0.0;
    for (int i = 0; i < centroid->size(); i++)
    {
        dist += pow(centroid->at(i) - point->get_feature_vector()->at(i), 2);
    }
    return sqrt(dist);
}

double kmeans::validate()
{
    double num_correct = 0.0;
    for (auto query_pt : *val_set)
    {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int i = 0; i < clusters->size(); i++)
        {
            double dist = euclidean_distance(clusters->at(i)->centroid, query_pt);
            if (dist < min_dist)
            {
                min_dist = dist;
                best_cluster = i;
            }
        }
        if (clusters->at(best_cluster)->most_freq_class == query_pt->get_label())
        {
            num_correct++;
        }
    }
    return 100.0 * (num_correct / (double)val_set->size());
}

double kmeans::test()
{
    double num_correct = 0.0;
    for (auto query_pt : *test_set)
    {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int i = 0; i < clusters->size(); i++)
        {
            double dist = euclidean_distance(clusters->at(i)->centroid, query_pt);
            if (dist < min_dist)
            {
                min_dist = dist;
                best_cluster = i;
            }
        }
        if (clusters->at(best_cluster)->most_freq_class == query_pt->get_label())
        {
            num_correct++;
        }
    }
    return 100.0 * (num_correct / (double)test_set->size());
}

int main()
{
    data_handler *dh = new data_handler();
    dh->read_feature_vector("../train-images.idx3-ubyte");
    dh->read_feature_labels("../train-labels.idx1-ubyte");
    dh->split_data();
    dh->count_classes();
    double perf = 0.0;
    double best_perf = 0.0;
    int best_k = 0;
    for (int k = dh->get_class_counts(); k < dh->get_training_data()->size() * 0.1; k++)
    {
        kmeans *kmeans_classifier = new kmeans(k);
        kmeans_classifier->set_training_set(dh->get_training_data());
        kmeans_classifier->set_test_set(dh->get_testing_data());
        kmeans_classifier->set_validation_set(dh->get_validation_data());
        kmeans_classifier->init_clusters();
        kmeans_classifier->train();
        perf = kmeans_classifier->validate();
        printf("Current Performance @ K = %d: %.3f %%\n", k, perf);
        if (perf > best_perf)
        {
            best_perf = perf;
            best_k = k;
        }
    }
    kmeans *kmeans_classifier = new kmeans(best_k);
    kmeans_classifier->set_training_set(dh->get_training_data());
    kmeans_classifier->set_test_set(dh->get_testing_data());
    kmeans_classifier->set_validation_set(dh->get_validation_data());
    kmeans_classifier->init_clusters();
    perf = kmeans_classifier->test();
    printf("Testing Performance @ K = %d: %.3f %%\n", best_k, perf);
}