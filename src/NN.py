# CS-445/545 ML Final Project
# Neural Network for breast cancer detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

# the model is set up globally with functions to support the training and result generation

# give user options for training size
print("You may train on 10%, 25%, or 100% of training data.")
train_size = input("Enter '10', '25', or '100': ")

while(train_size not in ['10','25','100']):
    train_size = input("Enter '10', '25', or '100': ")

# number of hidden nodes, learning rate, momentum, number of epochs
n = 20
lr = 0.01
momentum = 0.9
epochs = 100

# get the data
train_data = pd.read_csv(f'src/data/training{train_size}.csv').to_numpy()
test_data = pd.read_csv(f'src/data/test{train_size}.csv').to_numpy()
train_len = len(train_data)
test_len = len(test_data)

# store labels
train_labels = np.empty(train_len)
test_labels = np.empty(test_len)

np.random.shuffle(train_data)
# reset first element to 1 (bias)
for i in range(train_len):
    train_labels[i] = train_data[i,0]
    train_data[i,0] = 1
for i in range(test_len):
    test_labels[i] = test_data[i,0]
    test_data[i,0] = 1

# last column is blank (the was csv was loaded)
train_data = train_data[:,:31]
test_data = test_data[:,:31]

# rescale inputs
for i in range(1, 31):
    # find max of this element
    maximum = 0
    for j in range(len(train_data)):
        maximum = max(maximum, train_data[j,i])
    for j in range(len(train_data)):
        train_data[j,i] = train_data[j,i] / maximum
    
    for j in range(len(test_data)):
        maximum = max(maximum, test_data[j,i])
    for j in range(len(test_data)):
        test_data[j,i] = test_data[j,i] / maximum

# weight matrices
i2h = np.random.rand(n, 31)/1000 -0.0005
i2h_prev = np.zeros((n, 31))
h2o = np.random.rand(2, n)/1000 -0.0005
h2o_prev = np.zeros((2, n))
h2o_bias = np.random.rand(2)/1000 -0.0005
h2o_bias_prev = np.zeros(2)


# truth arrays
truth = np.array([[1,0],[0,1]])

# apply sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# get the current train accuracy of the model
def get_train_accuracy():

    # propogate
    h_activation = sigmoid(np.dot(train_data, i2h.transpose()))
    o_activation = sigmoid(np.dot(h_activation, h2o.transpose()) + h2o_bias)

    # compare outputs to correct
    output = o_activation.argmax(1)
    correct = np.sum(output == train_labels)
    return correct / train_len

# get the current test accuracy of the model
def get_test_accuracy():

    # propogate
    h_activation = sigmoid(np.dot(test_data, i2h.transpose()))
    o_activation = sigmoid(np.dot(h_activation, h2o.transpose()) + h2o_bias)

    # compare outputs to correct
    output = o_activation.argmax(1)
    correct = np.sum(output == test_labels)
    return correct / test_len

# generate the confusion matrix for the test data
def get_test_confusion():
    confusion = [[0,0],[0,0]]

    # propogate
    h_activation = sigmoid(np.dot(test_data, i2h.transpose()))
    o_activation = sigmoid(np.dot(h_activation, h2o.transpose()) + h2o_bias)

    # compare outputs to correct
    output = o_activation.argmax(1)

    for i in range(test_len):
        confusion[int(test_labels[i])][int(output[i])] += 1
    return confusion

# train on a single image
def train(i):
    global i2h, h2o, h2o_bias
    global i2h_prev, h2o_prev, h2o_bias_prev

    # propogate
    h_activation = sigmoid(np.dot(train_data[i], i2h.transpose()))
    #print(np.dot(h_activation, h2o.transpose()) + h2o_bias)
    o_activation = sigmoid(np.dot(h_activation, h2o.transpose()) + h2o_bias)

    # choose truth array
    true_val = truth[int(train_labels[i])]
    #print("True val:",true_val)
    #print("output activation:",o_activation)

    # back propogate
    o_error = o_activation * (1 - o_activation) * (true_val - o_activation)
    h_error = h_activation * (1 - h_activation) * np.dot(h2o.transpose(), o_error)

    # calculate weight changes
    h2o_delta = lr * np.multiply.outer(o_error, h_activation)\
        + momentum * h2o_prev
    h2o_bias_delta = lr * o_error\
        + momentum * h2o_bias_prev
    i2h_delta = lr * np.multiply.outer(h_error, train_data[i])\
        + momentum * i2h_prev

    # update weights
    h2o = h2o + h2o_delta
    h2o_bias = h2o_bias + h2o_bias_delta
    i2h = i2h + i2h_delta

    #print(h2o_delta)
    #input()
    # update previous
    h2o_prev = h2o_delta
    h2o_bias_prev = h2o_bias_delta
    i2h_prev = i2h_delta

train_acc_data = []
test_acc_data = []

train_acc_data.append(get_train_accuracy())
test_acc_data.append(get_test_accuracy())

# run the training
for j in range(epochs):
    for i in range(len(train_data)):
        train(i)

    train_acc_data.append(get_train_accuracy())
    test_acc_data.append(get_test_accuracy())

print("Final train accuracy:", train_acc_data[len(train_acc_data) - 1])
print("Final test accuracy:", test_acc_data[len(test_acc_data) - 1])

# generate plot
plt.figure(figsize= (10, 5))
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(range(0,epochs+1), train_acc_data, color="blue", label="Training data")
plt.plot(range(0,epochs+1), test_acc_data, color="red", label="Test data")
plt.legend()
plt.show()

# Testing data confusion matrix
plt.figure(figsize = (10, 6))
plt.title("Test Data Confusion Matrix")
seaborn.set(font_scale = 1)
seaborn.heatmap(get_test_confusion(), cmap="Blues", linewidth=0.5, annot=True, cbar_kws={'label': 'Frequency'})
plt.show()