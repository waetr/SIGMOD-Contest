Before run, make it:

    cmake .
    make

If you want to run the whole algorithm:

    ./test/test_faiss ../data_SIGMOD/contest-data-release-10m.bin output.bin 200 7 20 320

Note that only the dataset with 10M size can execute successfully, since the faiss will return -1 when the size of dataset is small.

If you want to run NN-Descent to adjust the parameters:

At the first time, run with the following command:

    ./tests/test_faiss_2 ../data_SIGMOD/contest-data-release-10m.bin rep/knng.bin

Then the program will generate *knng.bin* in the *rep/* path.

After that, run with the following command:

    ./tests/test_nndescent_refine ../data_SIGMOD/contest-data-release-10m.bin knng.bin output.bin 100 200 7 20 320

Meanings of these commands:

    100 -- K in KNNG (No need to modify)
    200 -- pool size
    20  -- sampling rate
    300 -- maximum size of nn_new and nn_old