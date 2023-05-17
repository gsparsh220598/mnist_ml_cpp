#! /bin/bash

# This script is used to add a new model to the project.
if [[ -z $MNIST_ML_CPP_ROOT ]]; then
    echo "Please define MNIST_ML_CPP_ROOT in your environment."
    exit 1
fi

dir=$(echo "$@" | tr a-z A-Z) # Convert to upper case
model_name_lower=$(echo "$@" | tr A-Z a-z) # Convert to lower case, $@ is the first argument

mkdir -p $MNIST_ML_CPP_ROOT/$dir/include $MNIST_ML_CPP_ROOT/$dir/src
touch $MNIST_ML_CPP_ROOT/$dir/Makefile
touch $MNIST_ML_CPP_ROOT/$dir/include/"$model_name_lower.hpp"
touch $MNIST_ML_CPP_ROOT/$dir/src/"$model_name_lower.cc"