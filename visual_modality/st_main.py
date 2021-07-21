# Recognizing Emotions evoked by Movies using Multitask Learning #
# Hassan Hayat, Carles Ventura, Agata Lapedriza #
# This script contains the training, validation, and test the SingleModality-SingleTask (ST-Visual) model using cross-validation sets for all availabel viewers # 

# import libraries 
import tensorflow as tf
import numpy as np
import pandas as pd
from random import shuffle
import csv
import os
import utils
import params

# all movies names 
movie_names = ["BMI","CHI","CRA","DEP","FNE","GLA","LOR"]
# all annotators 
all_viewers = ["viewer_1","viewer_2","viewer_3","viewer_4","viewer_5","viewer_6","viewer_7","avg_all_viewers"]
# create cross validation sets 
all_sets = utils.create_cross_validation_sets(movie_names)
# datasets for train,val and test with all necessary preprocessing 
# select viewer 
select_viewer = 0 # select from 0 to 7. 
single_viewer = all_viewers[select_viewer]
select_set_no = 0  # select between 0 and 6
# load data and divide in to train,val and test for single set of cross-validation
train_data, train_label, val_data, val_label, test_data, test_label = utils.create_train_val_test_datasets_st(all_sets,select_set_no,single_viewer)

# Multilayer perceptron
def MLP(input_tensor):
    # Flatten layer
    flatten_layer = tf.contrib.layers.flatten(input_tensor)
    # fully connected 
    fc1 = tf.contrib.layers.fully_connected(flatten_layer, 1024, activation_fn=tf.nn.relu,
    weights_initializer=tf.random_normal_initializer(seed=10),
    biases_initializer=tf.zeros_initializer(),
    trainable=True)
    # fully connected     
    fc2 = tf.contrib.layers.fully_connected(fc1, 512, activation_fn=tf.nn.relu,
    weights_initializer=tf.random_normal_initializer(seed=10),
    biases_initializer=tf.zeros_initializer(),
    trainable=True)
    # dropout
    dp1 = tf.nn.dropout(fc2, keep_prob)
    # fully connected
    fc3 = tf.contrib.layers.fully_connected(dp1, 256, activation_fn=tf.nn.relu,
    weights_initializer=tf.random_normal_initializer(seed=10),
    biases_initializer=tf.zeros_initializer(),
    trainable=True)
    # dropout
    dp2 = tf.nn.dropout(fc3, keep_prob)
    # output   
    output = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
    weights_initializer=tf.random_normal_initializer(seed=10),
    biases_initializer=tf.zeros_initializer(),
    trainable=True)
  
    return output

# inputs and labels 
x = tf.placeholder(tf.float32, shape=([None,1024]))
y = tf.placeholder(tf.float32, shape=([None,1]))
keep_prob = tf.placeholder(tf.float32, shape=())
weight = tf.placeholder(tf.float32, shape=())
# construct MLP 
logits = MLP(x,keep_prob)
# cross-entropy label loss
xnet = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits, pos_weight=weight))
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_constant = 0.2  # Choose an appropriate one.
loss_op = xnet + (reg_constant * sum((reg_losses)))
optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
train = optimizer.minimize(loss)
# model saver initialization 
saver = tf.train.Saver()
# start session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    check_loss = []
    patience_cnt = 0
    for epoch in range(params.EPOCHS):
        print('EPOCH #', epoch)
        # training 
        samples, labels = utils.get_shuffle_batch(train_data,train_label)
        neg_count = len(list(filter(lambda x: (x < 1), labels)))
        pos_count = len(list(filter(lambda x: (x > 0), labels)))
        training_weight = neg_count/pos_count
        train_loss = []
        val_loss = []
        targets = []
        md_preds = []
        for k in range(0,len(samples),params.batch_size):
            if k == int(len(samples)/params.batch_size)*params.batch_size:
              input_x = samples[k:]
              input_y = labels[k:]
              input_y = np.reshape(input_y,[len(input_y),1])
            else:
              input_x = samples[k:k+params.batch_size]
              input_y = labels[k:k+params.batch_size]
              input_y = np.reshape(input_y,[params.batch_size,1])
            _,loss,_ = sess.run([logits,loss_op,train], feed_dict={ x:input_x, y:input_y, keep_prob:0.6, weight:training_weight})
            train_loss.append(loss)
        print('Training Loss:',np.mean(train_loss))
        # validation  
        samples, labels = utils.get_shuffle_batch(val_data,val_label)
        neg_count = len(list(filter(lambda x: (x < 1), labels))) 
        pos_count = len(list(filter(lambda x: (x > 0), labels)))
        val_weight = neg_count/pos_count
        for l in range(0,len(samples),params.batch_size):
          if l == int(len(samples)/params.batch_size)*params.batch_size:
            input_x = samples[l:]
            input_y = labels[l:]
            targets.append(input_y)
            input_y = np.reshape(input_y,[len(input_y),1])
          else:
            input_x = samples[l:l+params.batch_size]
            input_y = labels[l:l+params.batch_size]
            targets.append(input_y)
            input_y = np.reshape(input_y,[params.batch_size,1])
          output, loss = sess.run([logits,loss_op], feed_dict={ x:input_x, y:input_y, keep_prob:1.0, weight:val_weight})
          val_loss.append(loss)
          md_preds.append(output)
        check_loss.append(np.mean(val_loss))
        print('Validation Loss',np.mean(val_loss))
        # early stopping
        patience = 20
        if epoch == 0:
          best_loss = check_loss[epoch]
        if epoch > 0 and check_loss[epoch] < best_loss:
            best_loss = check_loss[epoch]
            patience_cnt = 0
            save_path = "./save/" + "cnn_model"
            saver.save(sess, save_path)
        else:
            patience_cnt += 1
        if patience_cnt > patience:
            print("early stopping...")
            break
# testing
# load model
load_path = "./save/" + "cnn_model.meta"
saver = tf.train.import_meta_graph(load_path)
# start session
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint("./save/"))
    print("Model restored.")
    test_gt = []
    test_md = []
    samples, labels = utils.get_shuffle_batch(test_data,test_label)
    neg_count = len(list(filter(lambda x: (x < 1), labels)))
    pos_count = len(list(filter(lambda x: (x > 0), labels)))
    test_weight = neg_count/pos_count
    for k in range(0,len(samples),params.batch_size):
        if k == int(len(samples)/params.batch_size)*params.batch_size:
            input_x = samples[k:]
            input_y = labels[k:]
            input_y = np.reshape(input_y,[len(input_y),1])
            test_gt.append(input_y)
        else:
            input_x = samples[k:k+params.batch_size]
            input_y = labels[k:k+params.batch_size]
            input_y = np.reshape(input_y,[params.batch_size,1])
            test_gt.append(input_y)
        output = sess.run([logits], feed_dict={ x:input_x, y:input_y, keep_prob:1.0, weight:test_weight})
        test_md.append(output)
    test_md = np.squeeze(test_md)
    test_md = utils.get_vector(test_md)
    test_gt = utils.get_vector(test_gt)
    # binary 
    test_md = utils.get_binary(test_md)
    print('Test Results:',utils.md_performance(test_gt,test_md))


