import numpy as np
import tensorflow as tf
# import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image

def Load_data():
    # Lists for containing input training data
    data = []
    labels = []

    for i in range(num_classes):
        path = f"./gtsrb-german-traffic-sign/Train/{i}/"
        Class = os.listdir(path)
        for im in Class:
            try:
                image = cv2.imread(path + im)
                image_from_array = Image.fromarray(image, 'RGB')
                size_image = image_from_array.resize((30, 30))
                data.append(np.array(size_image))
                labels.append(i)
            except AttributeError:
                print("  ")

    print(f"Data length: {len(data)}")

    Input_images = np.array(data)
    Input_labels = np.array(labels)

    # Saving arrays to speed up upcoming runs
    np.save("Input_images", Input_images)
    np.save("Input_labels", Input_labels)

    return Input_images, Input_labels

# Defining reset_graph with set seed, to make runs consistent for testing
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# Declaration of basic parameters
width = 30
height = 30
num_channels = 3
num_classes = 43
input_shape = [width, height, num_channels]
dim_inputs = width*height*num_channels

# Loading data from npy file, or the original images
try:
    Input_images=np.load("Input_images.npy")
    Input_labels=np.load("Input_labels.npy")
except FileNotFoundError:
    print("Numpy files haven't been generated, or they are corrupted, creating them now.")
    Input_images, Input_labels = Load_data()

print(f"Shape of input images array: {Input_images.shape}")

# Shuffling the data before splitting into train and validation sets to avoid them containing images from only one or two classes
shuffle = np.arange(Input_images.shape[0])
np.random.seed(num_classes)
np.random.shuffle(shuffle)
Input_images = Input_images[shuffle]
Input_labels = Input_labels[shuffle]

# Splitting input data into training and validation sets (I only want to use the test set as the true final test.)
len_data = len(Input_images)
X_train = Input_images[:int(0.8 * len_data)]
X_test = Input_images[int(0.8 * len_data):]
Y_train = Input_labels[:int(0.8 * len_data)]
Y_test = Input_labels[int(0.8 * len_data):]

# Converting values to float between 0 and 1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
print(f"Train set length: {len(X_train)}, Test set length: {len(X_test)}")
# print(f"Train labels shape: {Train_labels.shape}")

# Flattening images to one vector containing all the pixels of all layers
print(f"Train set shape before flattening: {X_train.shape}")
X_train_flat = X_train.reshape(-1, dim_inputs)
print(f"Train set shape after flattening: {X_train_flat.shape}")
X_test_flat = X_test.reshape(-1, dim_inputs)

# Defining variables of the network and its layers
conv1_feature_maps = 4
conv1_kernel_size = [5, 5]
conv1_stride = 1
conv1_padding = "SAME"

conv2_feature_maps = 8
conv2_kernel_size = [3, 3]
conv2_stride = 2
conv2_padding = "SAME"

pool3_feature_maps = conv2_feature_maps

# Unused variables left for easier understandability
# num_neurons_fully_connected1 = 32
# num_neurons_output = 43

num_kernels = [input_shape[-1], 4, 8]
num_neurons = [32, num_classes]

reset_graph()
with tf.name_scope("inputs"):
    X_flat = tf.placeholder(tf.float32, [None, dim_inputs])
    X = tf.reshape(X_flat, shape=[-1] + input_shape)
    Y = tf.placeholder(tf.int32, shape=[None])

print(f"Current input shape: {X}")
print(f"kernel size: {conv1_kernel_size + [num_kernels[0], num_kernels[1]]}")

with tf.variable_scope("conv_1"):
    kernel = tf.get_variable("kernel", conv1_kernel_size + [num_kernels[0], num_kernels[1]])
    bias = tf.get_variable("bias", num_kernels[1])

    conv_result = tf.nn.conv2d(X, kernel, strides=conv1_stride, padding=conv1_padding)
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
    weights = tf.get_variable('weights', [int(fc1.get_shape()[-1]), num_neurons[1]])
    bias = tf.get_variable('bias', [num_classes], initializer=tf.constant_initializer(0.0))

    result = fc1 @ weights
    result = tf.add(result, bias)
    output = tf.nn.softmax(result)

print(f"Output: {output}")

with tf.name_scope("Loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=Y)
    loss = tf.reduce_mean(xentropy)

with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer().minimize(loss)


print(f"Y_train shape: {Y}, result shape: {result}")
with tf.name_scope("Accuracy"):
    # predictions = tf.equal(tf.math.argmax(Y_train, 1), tf.math.argmax(result, 1))
    predictions = tf.math.in_top_k(result, Y, 1)
    accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))

with tf.name_scope("Init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


def shuffle_batch(input, labels, batch_size):
    rnd_idx = np.random.permutation(len(input))
    num_batches = len(input) // batch_size
    for batch_idx in np.array_split(rnd_idx, num_batches):
        X_batch, Y_batch = input[batch_idx], labels[batch_idx]
        yield X_batch, Y_batch

num_epochs = 10
batch_size = 32

acc_train = []
acc_val = []

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1, num_epochs+1):
        next = shuffle_batch(X_train_flat, Y_train, batch_size)
        for i, (next_data, next_label) in enumerate(next):
            _, l, acc = sess.run([optimizer, loss, accuracy], feed_dict={X_flat: next_data, Y: next_label})
            # if i % 100 == 0:
                # acc_train.append(acc)
                # print(f"Accuracy: {acc}, Loss: {l}")
        acc_batch = accuracy.eval(feed_dict={X_flat: next_data, Y: next_label})
        acc_train.append(acc_batch)
        acc_test = accuracy.eval(feed_dict={X_flat: X_test_flat, Y: Y_test})
        acc_val.append(acc_test)
        print(f"{epoch}. Last batch accuracy: {acc_batch} Test accuracy: {acc_test}")

    # final_prediction = output.eval(feed_dict={ X_flat: Test_set_flat})

# print(f"Final prediction: {final_prediction}")


plt.figure(0)
plt.plot(acc_train, label='training accuracy')
plt.plot(acc_val, label='validation accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc=4)
plt.show()

# Validation_images, Validation_labels
try:
    Validation_images=np.load("Validation_images.npy")
    Validation_labels=np.load("Validation_labels.npy")
except FileNotFoundError:
    print("Numpy files haven't been generated for validation sets, or they are corrupted, creating them now.")

    data = []
    labels = []

    path = "./gtsrb-german-traffic-sign/Test/"
    Class = os.listdir(path)
    for im in Class:
        try:
            image = cv2.imread(path + im)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((30, 30))
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            print("  ")

    print(f"Data length: {len(data)}")

    Validation_images = np.array(data)
    Validation_labels = np.array(labels)

    # Saving arrays to speed up upcoming runs
    np.save("Validation_images", Validation_images)
    np.save("Validation_labels", Validation_labels)