import numpy as np
import tensorflow as tf
# import pandas as pd
import matplotlib.pyplot as plt
import os
from statistics import mean
# import cv2
# from PIL import Image

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
conv1_feature_maps = 16
conv1_kernel_size = [5, 5]
conv1_stride = 1
conv1_padding = "SAME"

conv2_feature_maps = 32
conv2_kernel_size = [3, 3]
conv2_stride = 2
conv2_padding = "SAME"

conv3_feature_maps = 64
conv3_kernel_size = [3, 3]
conv3_stride = 1
conv3_padding = "SAME"

pool3_feature_maps = conv2_feature_maps

# Unused variables left for easier understandability
# num_neurons_fully_connected1 = 32
# num_neurons_output = 43

num_kernels = [input_shape[-1], conv1_feature_maps, conv2_feature_maps, conv3_feature_maps]
num_neurons = [32, num_classes]

reset_graph()
with tf.name_scope("inputs"):
    X_flat = tf.placeholder(tf.float32, [None, dim_inputs])
    X = tf.reshape(X_flat, shape=[-1] + input_shape)
    Y = tf.placeholder(tf.int32, shape=[None])
    keep_prob = tf.placeholder(tf.float32)

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

with tf.name_scope("drop1"):
    drop_out1 = tf.nn.dropout(pool3, keep_prob)


with tf.variable_scope("conv_3"):
    kernel = tf.get_variable("kernel", conv3_kernel_size + [num_kernels[2], num_kernels[3]])
    bias = tf.get_variable("bias", num_kernels[3])
    conv_result = tf.nn.conv2d(drop_out1, kernel, strides=conv3_stride, padding=conv3_padding)
    biased = tf.add(conv_result, bias)
    conv3 = tf.nn.relu(biased)

print(f"conv3: {conv3}")

with tf.name_scope("pool6"):
    pool6 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1,], padding="VALID")

with tf.name_scope("drop2"):
    drop_out2 = tf.nn.dropout(pool6, keep_prob)
    drop_out2_flat = tf.reshape(drop_out2, shape=[-1, conv3_feature_maps * 3 * 3])

with tf.variable_scope("fully_connected1"):
    weights = tf.get_variable('weights', [int(drop_out2_flat.get_shape()[-1]), num_neurons[0]])
    bias = tf.get_variable('bias', [num_neurons[0]], initializer=tf.constant_initializer(0.0))

    result = drop_out2_flat @ weights
    result = tf.add(result, bias)
    result = tf.nn.relu(result)
    fc1 = result

print(f"Fully connected 1: {fc1}")

with tf.name_scope("drop3"):
    drop_out3 = tf.nn.dropout(fc1, keep_prob)

with tf.variable_scope("fully_connected2"):
    weights = tf.get_variable('weights', [int(drop_out3.get_shape()[-1]), num_neurons[1]])
    bias = tf.get_variable('bias', [num_classes], initializer=tf.constant_initializer(0.0))

    result = drop_out3 @ weights
    result = tf.add(result, bias)
    output = tf.nn.softmax(result)

print(f"Result: {result}")
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

num_epochs = 20
batch_size = 64

acc_train = []
acc_val = []

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1, num_epochs+1):
        next = shuffle_batch(X_train_flat, Y_train, batch_size)
        acc_epoch = []
        for i, (next_data, next_label) in enumerate(next):
            _, l, acc = sess.run([optimizer, loss, accuracy], feed_dict={X_flat: next_data, Y: next_label, keep_prob:0.75})
            acc_epoch.append(acc)
        acc_train.append(mean(acc_epoch))
        acc_test = accuracy.eval(feed_dict={X_flat: X_test_flat, Y: Y_test, keep_prob:1})
        acc_val.append(acc_test)
        print(f"{epoch}. Last batch accuracy: {mean(acc_epoch)} Test accuracy: {acc_test}")
    save_state = saver.save(sess, 'my-model', global_step=1)


plt.figure(0)
plt.plot(acc_train, label='training accuracy')
plt.plot(acc_val, label='validation accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc=4)
plt.show()

# Loading Validation_images, Validation_labels
try:
    Final_test_images=np.load("Final_test_images.npy")
    Final_test_labels=np.load("Final_test_labels.npy")
except FileNotFoundError:
    print("Numpy files haven't been generated for validation sets, or they are corrupted, creating them now.")

    labels = pd.read_csv("./gtsrb-german-traffic-sign/Test.csv")
    paths = labels['Path'].as_matrix()
    Final_test_labels = labels['ClassId'].values
    data = []

    path = "./gtsrb-german-traffic-sign/Test/"
    # Class = os.listdir(path)
    for f in paths:
        try:
            image = cv2.imread(path+f.replace('Test/', ''))
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((width, height))
            data.append(np.array(size_image))
            # labels.append(i)
        except AttributeError:
            print("  ")

    print(f"Data length: {len(data)}")

    Final_test_images = np.array(data)


    # Saving arrays to speed up upcoming runs
    np.save("Final_test_images", Final_test_images)
    np.save("Final_test_labels", Final_test_labels)

print(f"Shape of final test images array: {Final_test_images.shape}")
X_final_test = Final_test_images.astype('float32') / 255
X_final_test_flat = X_final_test.reshape(-1, dim_inputs)

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, save_state)

    acc_final_test = accuracy.eval(feed_dict={X_flat: X_final_test_flat, Y: Final_test_labels, keep_prob:1})
    print(f"Accuracy on test set: {acc_final_test}")