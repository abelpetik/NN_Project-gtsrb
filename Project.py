import numpy as np
import tensorflow as tf
# import pandas as pd
import matplotlib.pyplot as plt
import os
# import cv2
# from PIL import Image


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
    print("Numpy files haven't been generated, or they are corrupted, creating them now.")

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

print(Input_images.shape)

# Shuffling the data to avoid big batches containing images from only one or two classes
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
print(f"Train set length: {len(Train_set)}, Test set length: {len(Test_set)}")
print(f"Train labels shape: {Train_labels.shape}")

# Reshaping images to one vector containing all the pixels of all layers
print(Train_set.shape)
Train_set_flat = Train_set.reshape(-1, dim_inputs)
print(Train_set_flat.shape)
print(len(Train_set_flat))

Test_set_flat = Test_set.reshape(-1, dim_inputs)
print(Test_set_flat.shape)


# Defining reset_graph with set seed, to make runs consistent for testing
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
with tf.name_scope("inputs"):
    X_flat = tf.placeholder(tf.float32, [None, dim_inputs])
    X_train = tf.reshape(X_flat, shape=[-1] + input_shape)
    Y_train = tf.placeholder(tf.int32, shape=[None])

print(f"Current input shape: {X_train}")
print(f"kernel size: {conv1_kernel_size + [num_kernels[0], num_kernels[1]]}")

with tf.variable_scope("conv_1"):
    kernel = tf.get_variable("kernel", conv1_kernel_size + [num_kernels[0], num_kernels[1]])
    bias = tf.get_variable("bias", num_kernels[1])

    conv_result = tf.nn.conv2d(X_train, kernel, strides=conv1_stride, padding=conv1_padding)
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

print(f"pool3 output: {pool3}")
print(f"pool3 flat: {pool3_flat}")

print(f"weights size: {[int(pool3_flat.get_shape()[-1]), num_neurons[0]]}")

with tf.variable_scope("fully_connected1"):
    weights = tf.get_variable('weights', [int(pool3_flat.get_shape()[-1]), num_neurons[0]])
    bias = tf.get_variable('bias', [num_neurons[0]], initializer=tf.constant_initializer(0.0))

    result = pool3_flat @ weights
    result = tf.add(result, bias)
    result = tf.nn.relu(result)
    fc1 = result

print(f"Fully connected 1: {fc1}")

with tf.variable_scope("fully_connected2"):
    weights = tf.get_variable('weigths', [int(fc1.get_shape()[-1]), num_classes])
    bias = tf.get_variable('bias', [num_classes], initializer=tf.constant_initializer(0.0))

    result = fc1 @ weights
    result = tf.add(result, bias)
    output = tf.nn.softmax(result)

print(f"Output: {output}")

with tf.name_scope("Loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=Y_train)
    loss = tf.reduce_mean(xentropy)

with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer().minimize(loss)


print(f"Y_train shape: {Y_train}, result shape: {result}")
with tf.name_scope("Accuracy"):
    # predictions = tf.equal(tf.math.argmax(Y_train, 1), tf.math.argmax(result, 1))
    predictions = tf.math.in_top_k(result, Y_train, 1)
    accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))

with tf.name_scope("Init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

'''
def get_next_batch(indata, label, batch_length):
  used_in_batch = random.sample(range(indata.shape[0]), batch_length)
  batch_x = indata[used_in_batch, :]
  batch_y = label[used_in_batch]

  return batch_x, batch_y
'''

def shuffle_batch(input, labels, batch_size):
    rnd_idx = np.random.permutation(len(input))
    num_batches = len(input) // batch_size
    for batch_idx in np.array_split(rnd_idx, num_batches):
        X_batch, Y_batch = input[batch_idx], labels[batch_idx]
        yield X_batch, Y_batch

num_epochs = 10
num_iterations = 1000
batch_size = 32
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(num_iterations):
        next_data, next_label = get_next_batch(Train_set, Train_labels, batch_size)
        _, l, acc = sess.run([optimizer, loss, accuracy], feed_dict={X_train: next_data, Y_train: next_label})

        if i % 200 == 0:
            print(f"Step: {i}, loss: {l}, accuracy: {acc}")

        if i % 500 == 0:
            next_testdata, next_testlabel = get_next_batch(Test_set, Test_labels, batch_size)
            _, ac, out = sess.run([optimizer, accuracy, output], feed_dict={X_train: next_testdata, Y_train: next_testlabel})
            print(f"Accuracy: {ac}")
'''

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        for next_data, next_label in shuffle_batch(Train_set_flat, Train_labels, batch_size):
            _, l, acc = sess.run([optimizer, loss, accuracy], feed_dict={X_flat: next_data, Y_train: next_label})
            # print(f"Accuracy: {acc}, Loss: {l}")
        acc_batch = accuracy.eval(feed_dict={X_flat: next_data, Y_train: next_label})
        acc_test = accuracy.eval(feed_dict={X_flat: Test_set_flat, Y_train: Test_labels})
        print(f"{epoch}. Last batch accuracy: {acc_batch} Test accuracy: {acc_test}")

