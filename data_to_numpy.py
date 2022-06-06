# this file contains the function that reads all the training and test
# csv files and returns two np arrays, one with all the training examples
# and one with all the test examples
import numpy as np
import pandas as pd
import os

TOTAL_TRAIN_SIZE = 64452
TOTAL_TEST_SIZE = 21484

# returns [np.array(training_examples), np.array(test_examples)]
def data_to_numpy(path = ""):
    
    # training examples
    training_examples = np.empty((TOTAL_TRAIN_SIZE, 1876))

    i = 0
    # read each training csv file and write to one csv
    for filename_index in range(65):
        current_section = pd.read_csv(f"Training/Training{filename_index}.csv").to_numpy()
        # copy over the rows
        for j in range(len(current_section)):
            training_examples[(i * 1000) + j] = current_section[j]
        i += 1
    
    # test examples
    test_examples = np.empty((TOTAL_TEST_SIZE, 1876))

    i = 0
    # read each test csv file and write to one csv
    for filename_index in range(22):
        current_section = pd.read_csv(f"Test/Test{filename_index}.csv").to_numpy()
        # copy over the rows
        for j in range(len(current_section)):
            test_examples[(i * 1000) + j] = current_section[j]
        i += 1

    return training_examples, test_examples