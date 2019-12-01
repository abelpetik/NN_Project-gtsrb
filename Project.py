import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image


# Declaration of basic parameters
width = 30
height = 30
num_channels = 3
num_classes = 43
input_shape = [width, height, num_channels]
dim_inputs = width*height*num_channels

try:
    Input_images=np.load("Input_images.npy")
    Input_labels=np.load("Input_labels.npy")
except FileNotFoundError:
    print("Numpy files haven't been generated, creating them now.")

    # Lists for containing input training data
    data = []
    labels = []

    for i in range(num_classes):
        path=f"./gtsrb-german-traffic-sign/Train/{i}/"
        Class=os.listdir(path)
        for im in Class:
            try:
                image = cv2.imread(path+im)
                image_from_array = Image.fromarray(image, 'RGB')
                size_image = image_from_array.resize((30, 30))
                data.append(np.array(size_image))
                labels.append(i)
            except AttributeError:
                print("  ")

    print(f"Data length: {len(data)}")

    Input_images=np.array(data)
    Input_labels=np.array(labels)

    # Saving arrays to speed up upcoming runs
    np.save("Input_images", Input_images)
    np.save("Input_labels", Input_labels)

print(Input_labels.shape)

# Shuffling the test images to avoid big batches containing images from only one or two classes
shuffle = np.arange(Input_images.shape[0])
np.random.seed(num_classes)
np.random.shuffle(shuffle)
Input_images = Input_images[shuffle]
Input_labels = Input_labels[shuffle]

# Splitting input data into training and validation sets (I only want to use the test set as the true final test.)
len_data = len(Input_images)
Train_set = Input_images[:int(0.8*len_data)]
Test_set = Input_images[int(0.8*len_data):]
Train_labels = Input_labels[:int(0.8*len_data)]
Test_labels = Input_labels[int(0.8*len_data):]

# Converting values to float between 0 and 1
Train_set = Train_set.astype('float32')/255
Test_set = Test_set.astype('float32')/255
len_train = len(Train_set)
len_test = len(Test_set)
print(f"Train set length: {len_train}, Test set length: {len_test}")

'''
# Reshaping images to one vector containing all the pixels of all layers
print(Train_set.shape)
Train_set = Train_set.reshape(-1, dim_inputs)
print(Train_set.shape)
print(len(Train_set))

Test_set = Test_set.reshape(-1, dim_inputs)
print(Test_set.shape)
'''

# Defining reset_graph with set seed, to make runs consistent
def reset_graph(seed=num_classes):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

conv1_feature_maps = 4
conv1_kernel_size = [5, 5]
conv1_stride = 1
conv1_padding = "SAME"

conv2_feature_maps = 8
conv2_kernel_size = [3, 3]
conv2_stride = 2
conv2_padding = "SAME"

pool3_feature_maps = conv2_feature_maps

num_fully_connected1 = 32
num_output = num_classes

num_kernels = [input_shape[-1], 4, 8]
num_neurons = [32]

reset_graph()

X_train = tf.placeholder(tf.float32, [None] + input_shape)
current_input = X_train

print(X_train)
print(f"kernel size: {conv1_kernel_size + [num_kernels[0], num_kernels[1]]}")

with tf.variable_scope("conv_1"):
    kernel = tf.get_variable("kernel", conv1_kernel_size + [num_kernels[0], num_kernels[1]])
    bias = tf.get_variable("bias", num_kernels[1])

    conv_result = tf.nn.conv2d(current_input, kernel, strides=conv1_stride, padding=conv1_padding)
    biased = tf.add(conv_result, bias)
    conv1 = tf.nn.relu(biased)

print(conv1)

with tf.variable_scope("conv_2"):
    kernel = tf.get_variable("kernel", conv2_kernel_size + [num_kernels[1], num_kernels[2]])
    bias = tf.get_variable("bias", num_kernels[2])

    conv_result = tf.nn.conv2d(conv1, kernel, strides=conv2_stride, padding=conv2_padding)
    biased = tf.add(conv_result, bias)
    conv2 = tf.nn.relu(biased)

print(conv2)

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1,], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_feature_maps * 7 * 7])

print(pool3)
print(pool3_flat)

print(f"weights size: {[int(pool3_flat.get_shape()[-1]), num_neurons[0]]}")

with tf.name_scope("fully_connected1"):
    weights = tf.get_variable('weights', [int(pool3_flat.get_shape()[-1]), num_neurons[0]])
    bias = tf.get_variable('bias', [num_neurons[0]], initializer=tf.constant_initializer(0.0))

    result = pool3_flat @ weights
    result += bias
    result = tf.nn.relu(result)

print(result)
