CC=g++
SRC := $(MNIST_ML_CPP_ROOT)/KMEANS/src
CFLAGS := -std=c++11
INCLUDE_DIR := $(PWD)/include/*

all: main

main: $(SRC)/kmeans.cc
	$(CC) $(CFLAGS) $(SRC)/kmeans.cc -o main -L$(MNIST_ML_CPP_ROOT)/lib -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_CPP_ROOT)/include -ldata

clean:
	rm main