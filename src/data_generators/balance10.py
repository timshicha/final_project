
import numpy as np
import pandas as pd

# 'B' = 0
# 'M' = 1
data = pd.read_csv("src/data/data.csv").to_numpy()[:,1:]

for i in range(len(data)):
    if(data[i,0] == 'B'):
        data[i,0] = 0
    else:
        data[i,0] = 1

np.random.shuffle(data)

# first 100 will be validation set
test_data = data[:100,:]

# 47 of 470 (10%) training examples
train_data = data[100:147,:]
train_len = len(train_data)

# create basket into which we will store new train data
new_train_data = np.empty((10000,len(train_data[0,:])))
index = 0

# for each train example
for i in range(train_len):

    # devide if we'll make 4 or 7 duplicates
    duplicates = 4
    if(train_data[i,0] == 1):
        duplicates = 7

    # create the duplicates
    for j in range(duplicates):
        new_train_data[index] = train_data[i]
        index += 1
new_train_data = new_train_data[:index,:]

# shuffle train data
np.random.shuffle(new_train_data)

# that's our new data
m = 0
b = 0
for i in range(len(new_train_data)):
    if(new_train_data[i,0] == 0):
        m += 1
    else:
        b += 1
print("Training 'M' occurances: ", m)
print("Training 'B' occurances: ", b)
m = 0
b = 0
for i in range(len(test_data)):
    if(test_data[i,0] == 1):
        m += 1
    else:
        b += 1
print("Testing 'M' occurances: ", m)
print("Testing 'B' occurances:", b)

# store to files
test_dataframe = pd.DataFrame(test_data)
train_dataframe = pd.DataFrame(new_train_data)

# store to csv
train_dataframe.to_csv('src/data/training10.csv', index=False)
test_dataframe.to_csv('src/data/test10.csv', index=False)
