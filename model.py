import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

import code

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from parameters import *
from helper import *



def variable_creation(train_data):

# This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))
    train_all_data_node = tf.constant(train_data)

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
    

    
    tf.summary.histogram('conv1_weights 1', tf.reshape(conv1_weights, [-1]))
    
    prinnnn = tf.reshape(conv1_weights[:,:,:,0], [1, 5, 5, 3])
    

    tf.summary.image('summconv_visu', prinnnn, max_outputs=3)

    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    
    
    
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))


    all_params_node = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases]
    all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases']

    all_nodes = {       "conv1_w": conv1_weights, 
                        "conv1_b": conv1_biases, 
                        "conv2_w": conv2_weights, 
                        "conv2_b": conv2_biases, 
                        "fc1_w": fc1_weights,
                        "fc1_b": fc1_biases, 
                        "fc2_w": fc2_weights, 
                        "fc2_b": fc2_biases}

    
    return train_data_node, train_labels_node, train_all_data_node, all_params_node, all_params_names, all_nodes


def operation( train_data_node, train_labels_node, train_all_data_node, all_params_node, all_params_names, all_nodes, train_size ):
    
    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, all_nodes, True)  # BATCH_SIZE*NUM_LABELS
    # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())

    #mesurer l'ecart entre variable catégrique entre ta proba et la realité
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_labels_node,
                                                   logits=logits))

    tf.summary.scalar('loss', loss)

    
    all_grads_node = tf.gradients(loss, all_params_node)
    all_grad_norms_node = []
    
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.summary.scalar(all_params_names[i], norm_grad_i)
    
    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(all_nodes['fc1_w']) + tf.nn.l2_loss(all_nodes['fc1_b']) +
                    tf.nn.l2_loss(all_nodes['fc2_w']) + tf.nn.l2_loss(all_nodes['fc2_b']))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    # tf.scalar_summary('learning_rate', learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    
    
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.0).minimize(loss,
                                                         global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    train_all_prediction = tf.nn.softmax(model(train_all_data_node, all_nodes))



    return optimizer, loss, learning_rate, train_prediction
  



# We will replicate the model structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.
def model(data,  all_nodes, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        all_nodes['conv1_w'],
                        strides=[1, 1, 1, 1],
                        padding='SAME')

   
                       
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, all_nodes['conv1_b']))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    conv2 = tf.nn.conv2d(pool,
                            all_nodes['conv2_w'],
                            strides=[1, 1, 1, 1],
                            padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, all_nodes['conv2_b']))
    pool2 = tf.nn.max_pool(relu2,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    # Uncomment these lines to check the size of each layer
    # print 'data ' + str(data.get_shape())
    # print 'conv ' + str(conv.get_shape())
    # print 'relu ' + str(relu.get_shape())
    # print 'pool ' + str(pool.get_shape())
    # print 'pool2 ' + str(pool2.get_shape())

    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool2.get_shape().as_list()
    reshape = tf.reshape(
        pool2,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, all_nodes['fc1_w']) + all_nodes['fc1_b'])
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    #if train:
    #    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    out = tf.matmul(hidden, all_nodes['fc2_w'] ) + all_nodes['fc2_b']

    if train:
        summary_id = '_0'
        s_data = get_image_summary(data)
        tf.summary.image('summary_data' + summary_id, s_data, max_outputs=3)
        s_conv = get_image_summary(conv)
        tf.summary.image('summary_conv' + summary_id, s_conv, max_outputs=3)
        s_pool = get_image_summary(pool)
        tf.summary.image('summary_pool' + summary_id, s_pool, max_outputs=3)
        s_conv2 = get_image_summary(conv2)
        tf.summary.image('summary_conv2' + summary_id, s_conv2, max_outputs=3)
        s_pool2 = get_image_summary(pool2)
        tf.summary.image('summary_pool2' + summary_id, s_pool2, max_outputs=3)
    return out