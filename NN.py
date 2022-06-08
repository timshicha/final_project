
import pandas as pd
import numpy as np

from data_to_numpy import data_to_numpy


n = int(input("Number of hidden units: "))
lr = float(input("Learning rate: "))
momentum = float(input("Momentum: "))
epochs = int(input("Number of epochs: "))

# get the images
print("\nLoading image files...")
train_images, test_images = data_to_numpy()
print("Done.\n\nRunning NN...\n")
train_length = len(train_images)
test_length = len(test_images)

# shuffle training
np.random.shuffle(train_images)

# store labels
train_labels = train_images[:,0]
test_labels = test_images[:,0]

# Decrease weights from 0-255 to 0-1
train_images = np.divide(train_images, 255)
test_images = np.divide(test_images, 255)

# use first element for bias
for i in range(train_length):
    train_images[i,0] = 1
for i in range(test_length):
    test_images[i,0] = 1

# bias values for hidden to output
bias_values = np.array([1,1])

# weight matrices
input_to_hidden_weights = np.random.rand(n, 1876)
hidden_to_output_weights = np.random.rand(2, n)
hidden_to_output_bias = np.random.rand(2)

# previous weight changes for momentum
input_to_hidden_previous_change = np.zeros((n, 1876))
hidden_to_output_previous_change = np.zeros((2, n))
hidden_to_output_bias_previous_change = np.zeros(2)

# scale weights to be small
input_to_hidden_weights = np.divide(input_to_hidden_weights, 10)
input_to_hidden_weights = input_to_hidden_weights - 0.05
hidden_to_output_weights = np.divide(hidden_to_output_weights, 10)
hidden_to_output_weights = hidden_to_output_weights - 0.05
hidden_to_output_bias = np.divide(hidden_to_output_bias, 10)
hidden_to_output_bias = hidden_to_output_bias - 0.05

# truth arrays
truth = np.array([[0.9,0.1],[0.1,0.9]])

# sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Accuracy of training set
def training_set_accuracy():

    hidden_activation = sigmoid(np.dot(train_images, input_to_hidden_weights.transpose()))
    output_activation = sigmoid(np.dot(\
        hidden_activation, hidden_to_output_weights.transpose()) + np.dot(bias_values, hidden_to_output_bias))
    
    output = output_activation.argmax(1) # list of predictions
    correct = np.sum(output == train_labels) # count number of correct predictions

    return correct / train_length

# Accuracy of the testing set
def test_set_accuracy():

    hidden_activation = sigmoid(np.dot(test_images, input_to_hidden_weights.transpose()))
    output_activation = sigmoid(np.dot(\
        hidden_activation, hidden_to_output_weights.transpose()) + np.dot(bias_values, hidden_to_output_bias))

    output = output_activation.argmax(1)
    correct = np.sum(output == test_labels)

    return correct / test_length

def train_confusion():
    conf_matrix = np.zeros((2,2))
    hidden_activation = sigmoid(np.dot(train_images, input_to_hidden_weights.transpose()))
    output_activation = sigmoid(np.dot(\
        hidden_activation, hidden_to_output_weights.transpose()) + np.dot(bias_values, hidden_to_output_bias))
    
    output = output_activation.argmax(1) # list of predictions

    # place each image into matrix
    for i in range(train_length):
        conf_matrix[int(train_labels[i]), int(output[i])] += 1
    return conf_matrix

# train the model on an input
def train_on_image(image_index):
    global input_to_hidden_weights, hidden_to_output_weights, hidden_to_output_bias
    global input_to_hidden_previous_change, hidden_to_output_bias_previous_change
    global hidden_to_output_previous_change, hidden_to_output_bias_previous_change

    # progoate
    hidden_activation = sigmoid(np.dot(\
        train_images[image_index], input_to_hidden_weights.transpose()))
    output_activation = sigmoid(np.dot(\
        hidden_activation, hidden_to_output_weights.transpose()) + np.dot(bias_values, hidden_to_output_bias))

    # choose correct truth array
    true_value = truth[int(train_labels[image_index])]

    # back propogate
    output_error = output_activation * (1 - output_activation) * (true_value - output_activation)
    hidden_error = hidden_activation * (1 - hidden_activation) * np.dot(hidden_to_output_weights.transpose(), output_error)

    # calculate weight changes and update weights
    hidden_to_output_delta = lr * np.multiply.outer(output_error, hidden_activation) + momentum * hidden_to_output_previous_change
    hidden_to_output_weights = hidden_to_output_weights + hidden_to_output_delta

    hidden_to_output_bias_delta = lr * np.multiply.outer(output_error, bias_values) + momentum * hidden_to_output_bias_previous_change
    hidden_to_output_bias = hidden_to_output_bias + hidden_to_output_bias_delta

    input_to_hidden_delta = lr * np.multiply.outer(hidden_error, train_images[image_index]) + momentum * input_to_hidden_previous_change
    input_to_hidden_weights = input_to_hidden_weights + input_to_hidden_delta

    # store changes for momentum
    input_to_hidden_previous_change = input_to_hidden_delta
    hidden_to_output_bias_previous_change = hidden_to_output_bias_delta
    hidden_to_output_previous_change = hidden_to_output_delta

print("Starting test accuracy:", test_set_accuracy())
# train for x epochs
for i in range(epochs):
    # reset momentum
    input_to_hidden_previous_change = 0
    hidden_to_output_bias_previous_change = 0
    hidden_to_output_previous_change = 0

    # train on each example
    for j in range(train_length):
        train_on_image(j)

    # display accuracies
    print(f"Epoch {i+1}")
    print(training_set_accuracy(), test_set_accuracy())

print("Confusion matrix\n", train_confusion())
