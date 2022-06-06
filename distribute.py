# The purpose of this program is to convert the two large files of examples
# into smaller ones for performance and github limits


import numpy as np
import pandas as pd

def distribute_train():

    train_data = pd.read_csv("Training_set.csv").to_numpy()
    length = len(train_data)

    i = 0
    # store into files of 1000
    while((i * 1000) < (length - 1000)):
        to_store = train_data[(i*1000):(i+1)*1000,:]
        dataframe = pd.DataFrame(to_store)
        dataframe.to_csv(f"Training/Training{i}.csv", index=False)
        i += 1

    # store last n
    to_store = train_data[(i*1000):,:]
    dataframe = pd.DataFrame(to_store)
    dataframe.to_csv(f"Training/Training{i}.csv", index=False)

def distribute_test():

    test_data = pd.read_csv("Test_set.csv").to_numpy()
    length = len(test_data)

    i = 0
    # storing into files of 1000
    while((i * 1000) < (length - 1000)):
        to_store = test_data[(i*1000):(i+1)*1000,:]
        dataframe = pd.DataFrame(to_store)
        dataframe.to_csv(f"Test/Test{i}.csv", index=False)
        i += 1

    # store last n
    to_store = test_data[(i*1000):,:]
    dataframe = pd.DataFrame(to_store)
    dataframe.to_csv(f"Test/Test{i}.csv", index=False)

distribute_test()