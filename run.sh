#!/bin/bash

make clean
make
./tests/test_faiss tests/dummy-data.bin output.bin 200 7 20 300
