from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os, glob
import cv2
import random, shutil

tf.logging.set_verbosity (tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # LFW Dataset images are 60 * 60 pixels, and have one color channel
    input_layer = tf.reshape (features["x"], [-1, 60, 60, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height (60 * 60).
    # Input Tensor Shape: [batch_size, 60, 60, 1]
    # Output Tensor Shape: [batch_size, 60, 60, 32]
    conv1 = tf.layers.conv2d (inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 60, 60, 32]
    # Output Tensor Shape: [batch_size, 30, 30, 32]
    pool1 = tf.layers.max_pooling2d (inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 30, 30, 32]
    # Output Tensor Shape: [batch_size, 30, 30, 64]
    conv2 = tf.layers.conv2d (inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 3x3 filter and stride of 2
    # Input Tensor Shape: [batch_size, 30, 30, 64]
    # Output Tensor Shape: [batch_size, 10, 10, 64]
    pool2 = tf.layers.max_pooling2d (inputs=conv2, pool_size=[3, 3], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 10, 10, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 256]
    pool2_flat = tf.reshape (pool2, [-1, 7 * 7 * 256])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 256]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense (inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.25 probability
    dropout = tf.layers.dropout (inputs=dense, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 2]
    logits = tf.layers.dense (inputs=dropout, units=2)

    #y_pred = tf.nn.softmax (layer_fc2, name="y_pred")

    predictions = {  # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax (input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax (logits, name="softmax_tensor")}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec (mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot (indices=tf.cast (labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy (onehot_labels=onehot_labels, logits=logits)

    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer (learning_rate=0.0005)   # Learning rate = 0.0005
        train_op = optimizer.minimize (loss=loss, global_step=tf.train.get_global_step ())
        return tf.estimator.EstimatorSpec (mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec (mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# Main Function:
# The function to call all the phases required for the application of CNN model for training and testing on the Dataset
# Inputs: None
# Output: Accuracy on the test data
def main(unused_argv):
    mountain_bike_train_path = input ("Enter path of mountain bikes - training folder. \nFOR EXAMPLE: /Users/Aravind/Desktop/Nascent/training/mountain_bikes/  \nEnter just path WITHOUT quotes and WITHOUT missing the '/' mark at the end of the path:")
    road_bike_train_path = input ("Similarly enter path of road bikes - training folder. \nEnter Here:")
    mountain_bike_test_path = input ("Now the path of testing folder - mountain bikes: ")
    road_bike_test_path = input ("Wooh!!!!!! The last path - road bikes testing folder. Sit back and relax after entering! :")
    # Initialize total_accuracy and k fold values to 0 and 2(can be changed)
    total_val = 0
    k = 5
    for i in range(k):
        # Line 105 - 114: First check to see if both the test folders are empty
        # Line 105 - 108: First check to see if the mountain bike test folder is empty
        os.chdir(mountain_bike_test_path)   # set the testing directory as source.
        destination = mountain_bike_train_path  # set the training directory as destination
        for i in glob.glob('*.jpg'):   # go through each image file in the test folder, if present (all images are .jpg)
            shutil.move(i, destination)  # move each file to destination
        # Line 109 - 114: First check to see if the road bike test folder is empty
        os.chdir(road_bike_test_path)  # set the testing directory as source.
        destination = road_bike_train_path  # set the training directory as destination
        if glob.glob('*.jpg') != 0:   # check if the test folder is empty
            for i in glob.glob('*.jpg'):   # go through each image file in the test folder, if present (all images are .jpg)
                shutil.move(i, destination)   # move each file to destination
        # Line 116 - 123: Move the images from Training folder to Testing folder
        num_of_testing_images = random.randint(10, 20)     # Generates the random number of testing images (10 -20)
        random_num_in_training_images = random.sample(range(100), int(num_of_testing_images))   # Generates num_of_testing_images(from line 114) random numbers from 0 to 100
        # Loop to move random_num_in_training_images(from Line 115) number of images into testing folder from training folder
        for training_image_i in random_num_in_training_images:
            os.chdir (mountain_bike_train_path)
            os.rename(mountain_bike_train_path + "mountain_bike_"+ str(training_image_i)+ ".jpg", mountain_bike_test_path + "mountain_bike_"+ str(training_image_i)+ ".jpg")
            os.chdir (road_bike_train_path)
            os.rename(road_bike_train_path + "road_bike_"+ str(training_image_i)+ ".jpg",  road_bike_test_path + "road_bike_"+ str(training_image_i)+ ".jpg")
        # Line 125 - 136: Generate mountain bike train set array - line 135
        os.chdir(mountain_bike_train_path) # set mountain bike training path as current directory
        num_mountain_bikes_train = len (glob.glob ('*')) + 1   # number of mountain bikes in training folder
        train_mountain_bike = [[0] * 3600] * num_mountain_bikes_train  # list of lists to store flattened images in mountain bike training folder
        train_mountain_bike = np.array(train_mountain_bike)  # convert the list(from line 127) to array
        mountain_bike_number_train = 1   # set the mountain bike number to 1 every new execution
        for mountain_bike in glob.glob ('*.jpg'):  # go through every image in mountain bike training folder
            bike_image = cv2.imread (mountain_bike, 0)  # open each image in gray scale
            bike_image = cv2.resize(bike_image, (60,60), None, interpolation=cv2.INTER_NEAREST)   # resize image size to 60*60
            bike_image = cv2.equalizeHist (bike_image)   # Histogram equalization of the input image
            bike_image_flatten = bike_image.flatten ()    # flatten image from 60*60 to 3600
            train_mountain_bike[mountain_bike_number_train, :] = bike_image_flatten  # fill the mountain bike train array
            mountain_bike_number_train = mountain_bike_number_train + 1  # increment mountain bike number
        # Line 138 - 149: Generate road bike train set array - line 148
        os.chdir (road_bike_train_path)  # set road bike training path as current directory
        num_road_bikes_train = len (glob.glob('*')) + 1  # number of road bikes in training folder
        train_road_bike = [[0] * 3600] * num_road_bikes_train   # list of lists to store flattened images in road bike training folder
        train_road_bike = np.array (train_road_bike)    # convert the list(from line 140) to array
        road_bike_number_train = 1  # set the road bike number to 1 every new execution
        for road_bike in glob.glob ('*.jpg'):  # go through every image in road bike training folder
            bike_image = cv2.imread (road_bike, 0)    # open each image in gray scale
            bike_image = cv2.resize (bike_image, (60, 60), None, interpolation=cv2.INTER_NEAREST)   # resize image size to 60*60
            bike_image = cv2.equalizeHist (bike_image)   # Histogram equalization of the input image
            bike_image_flatten = bike_image.flatten ()    # flatten image from 60*60 to 3600
            train_road_bike[road_bike_number_train, :] = bike_image_flatten    # fill the road bike train array
            road_bike_number_train = road_bike_number_train + 1  # increment road bike number
        # Line 151 - 155: concatenate training data and training labels
        train_labels_mountain_bike = np.asarray (np.zeros (len (train_mountain_bike)), dtype=np.int32)
        train_labels_road_bike = np.asarray (np.ones (len (train_road_bike)), dtype=np.int32)
        train_data_bike = np.concatenate ((train_mountain_bike, train_road_bike), axis=0)
        train_data_bike = np.float32(train_data_bike)
        train_labels_bike = np.concatenate ((train_labels_mountain_bike, train_labels_road_bike), axis=0)
        # Line 157 - 168: process mountain bike images in test folder
        os.chdir (mountain_bike_test_path)  #
        num_mountain_bikes_test = len (glob.glob ('*')) + 1
        test_mountain_bike = [[0] * 3600] * num_mountain_bikes_test
        test_mountain_bike = np.array (test_mountain_bike)
        mountain_bike_number_test = 1
        for mountain_bike in glob.glob ('*.jpg'):
            bike_image = cv2.imread (mountain_bike, 0)
            bike_image = cv2.resize (bike_image, (60, 60), None, interpolation=cv2.INTER_NEAREST)
            bike_image = cv2.equalizeHist (bike_image)
            bike_image_flatten = bike_image.flatten()
            test_mountain_bike[mountain_bike_number_test, :] = bike_image_flatten
            mountain_bike_number_test = mountain_bike_number_test + 1
        # Line 170 - 181: process road bike images in test folder
        os.chdir (road_bike_test_path)
        num_road_bikes_test = len(glob.glob('*')) + 1
        test_road_bike = [[0] * 3600] * num_road_bikes_test
        test_road_bike = np.array (test_road_bike)
        road_bike_number_test = 1
        for road_bike in glob.glob ('*.jpg'):
            bike_image = cv2.imread (road_bike, 0)
            bike_image = cv2.resize (bike_image, (60, 60), None, interpolation=cv2.INTER_NEAREST)
            bike_image = cv2.equalizeHist (bike_image)
            bike_image_flatten = bike_image.flatten ()
            test_road_bike[road_bike_number_test, :] = bike_image_flatten
            road_bike_number_test = road_bike_number_test + 1
        # concatenate test data and test labels
        test_labels_mountain_bike = np.asarray (np.zeros (len (test_mountain_bike)), dtype=np.int32)
        test_labels_road_bike = np.asarray (np.ones (len (test_road_bike)), dtype=np.int32)
        eval_data_bike = np.concatenate ((test_mountain_bike, test_road_bike), axis=0)
        eval_data_bike = np.float32 (eval_data_bike)
        eval_labels_bike = np.concatenate ((test_labels_mountain_bike, test_labels_road_bike), axis=0)

        # Create the Estimator
        bike_classifier = tf.estimator.Estimator (model_fn=cnn_model_fn)

        # Set up logging for predictions
        # Log the values in the "Softmax" tensor with label "probabilities"
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook (tensors=tensors_to_log, every_n_iter=50)

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn (x={"x": train_data_bike}, y=train_labels_bike, batch_size=100,
                                                         num_epochs=500, shuffle=True)
        bike_classifier.train (input_fn=train_input_fn, steps=500, hooks=[logging_hook])

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data_bike}, y=eval_labels_bike, num_epochs=1,
                                                        shuffle=False)
        eval_results = bike_classifier.evaluate(input_fn=eval_input_fn)
        val = eval_results["accuracy"]

        # Accuracy across k values are added
        total_val = total_val + val
        # restore the images from test data folder into train data folder
        for training_image_i in random_num_in_training_images:
            os.chdir (mountain_bike_train_path)
            os.rename(mountain_bike_test_path + "mountain_bike_"+ str(training_image_i)+ ".jpg", mountain_bike_train_path + "mountain_bike_"+ str(training_image_i)+ ".jpg")
            os.chdir (road_bike_train_path)
            os.rename(road_bike_test_path + "road_bike_"+ str(training_image_i)+ ".jpg", road_bike_train_path + "road_bike_"+ str(training_image_i)+ ".jpg")
    print ("Accuracy: ", total_val / k)
if __name__ == '__main__':
    tf.app.run()