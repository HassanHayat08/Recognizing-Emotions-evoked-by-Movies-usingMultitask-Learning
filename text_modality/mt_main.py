# Multi-Model Multi-Task Learning for Emotion Recognition in Movies #
# Multitask learning using concatenation of the modalities #
# # This script contains the training, validation, and test the SingleModality-MultiTask (MT-Text) using cross-validation sets for all availabel viewers #

# import libraries 
import tensorflow as tf
import numpy as np
from random import shuffle
import re, string
import csv
import random 
import collections
import os
import utils 
import params

# all movie names
movie_names = ["BMI","CHI","CRA","DEP","FNE","GLA","LOR"]
# all annotators
all_annotators = ["experienced_1","experienced_2","experienced_3","experienced_4","experienced_5","experienced_6","experienced_7",'avg_experienced']
# create cross validation sets 
all_sets = utils.create_cross_validation_sets(movie_names)
# select annotator no, select from 0 to 7 
select_set = 0
single_set = all_sets[select_set]
train_mv_names = single_set["train_mv"]
val_mv_names = single_set["val_mv"]
test_mv_names = single_set["test_mv"]
# load data from all 7 annotators to create train, val and test data 
train_data = utils.load_data(train_mv_names,all_annotators)
val_data = utils.load_data(val_mv_names,all_annotators)
test_data = utils.load_data(test_mv_names,all_annotators)
# data cleaning 
train_samples, train_labels = utils.data_cleaning(train_data)
val_samples, val_labels = utils.data_cleaning(val_data)
test_samples, test_labels = utils.data_cleaning(test_data)
# get max length of sentence 
max_words = utils.max_length_of_sentence(train_samples,val_samples,test_samples)
# data padding 
train_samples = utils.data_padding(train_samples,max_words)
val_samples = utils.data_padding(val_samples,max_words)
test_samples = utils.data_padding(test_samples,max_words)
# create vocabulary 
vocabulary = utils.get_vocabulary(movie_names)
VOCAB_LEN = len(vocabulary)
# inputs and labels 
x = tf.placeholder(tf.int32, shape=([None,max_words]))
y = tf.placeholder(tf.float32, shape=([None,1]))
keep_prob = tf.placeholder(tf.float32, shape=())
weight = tf.placeholder(tf.float32, shape=())
tensor = tf.placeholder(tf.float32, shape=([None,1200]))
# Store bias
biases = {
    'bc1': tf.get_variable('bc1',shape=[300], initializer=tf.zeros_initializer()),
    'bc2': tf.get_variable('bc2',shape=[300], initializer=tf.zeros_initializer()),
    'bc3': tf.get_variable('bc3',shape=[300], initializer=tf.zeros_initializer()),
    }

# function for 2D convolution 
def conv2d(X, k_size, b):
    # Conv2D wrapper, with bias and relu activation 
    # VALID means No Padding and SAME means with padding 
    X = tf.layers.conv2d(inputs=X, filters=300, kernel_size=k_size, strides=[3,3], padding="SAME",
                         activation=None,
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                         trainable=True)
    X = tf.nn.bias_add(X, b)
    return tf.nn.relu(X)

# function for 2D max-pooling
def maxpool2d(X,k):
    # MaxPool2D wrapper
    return tf.nn.max_pool(X, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

# Create conv model
def conv_net(x, VOCAB_LEN, EMBED_SIZE, biases):
    with tf.variable_scope('conv'):
      # embedding layer 
      embed = tf.contrib.layers.embed_sequence(ids=x, vocab_size=VOCAB_LEN, embed_dim=EMBED_SIZE,
                    initializer=tf.truncated_normal_initializer(mean=0.0,stddev=1.0), regularizer=None,
                    trainable=True)
      net = tf.reshape(embed, shape=[-1, 18, EMBED_SIZE, 1])
      # convolution layer
      conv1 = conv2d(net, [2,EMBED_SIZE], biases['bc1'])
      # max-pooling (down-sampling)
      conv1 = maxpool2d(conv1, k=3)
      # convolution layer
      conv2 = conv2d(conv1, [3,EMBED_SIZE], biases['bc2'])
      # max-pooling (down-sampling)
      conv2 = maxpool2d(conv2, k=3)
      # flatten layer
      flatten_layer = tf.contrib.layers.flatten(conv2)
      return flatten_layer

# construct model 
conv_output = conv_net(x, VOCAB_LEN, params.EMBED_SIZE, biases)
# graph loss #
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) 
conv_var = [v for v in tf.trainable_variables() if v.name == "conv"]
reg_constant = 0.2 # Choose an appropriate one.
# personalize 1 
logits_1 = utils.personalized_model_1(tensor,keep_prob)
xnet_1 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_1, pos_weight=weight))
p_1_var = [v for v in tf.trainable_variables() if v.name == "p_1"]
reg_losses_1 = sum(conv_var) + sum(p_1_var)
loss_op_1 = xnet_1 + (reg_constant * reg_losses_1)
optimizer_1 = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_1 = optimizer_1.minimize(loss_op_1)
# personalize 2 
logits_2 = utils.personalized_model_2(tensor,keep_prob)
xnet_2 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_2, pos_weight=weight))
p_2_var = [v for v in tf.trainable_variables() if v.name == "p_2"]
reg_losses_2 = sum(conv_var) + sum(p_2_var)
loss_op_2 = xnet_2 + (reg_constant * sum(reg_losses))
optimizer_2 = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_2 = optimizer_2.minimize(loss_op_2)
# personalize 3 
logits_3 = utils.personalized_model_3(tensor,keep_prob)
xnet_3 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_3, pos_weight=weight))
p_3_var = [v for v in tf.trainable_variables() if v.name == "p_3"]
reg_losses_3 = sum(conv_var) + sum(p_3_var)
loss_op_3 = xnet_3 + (reg_constant * reg_losses_3)
optimizer_3 = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_3 = optimizer_3.minimize(loss_op_3)
# personalize 4 
logits_4 = utils.personalized_model_4(tensor,keep_prob)
xnet_4 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_4, pos_weight=weight))
p_4_var = [v for v in tf.trainable_variables() if v.name == "p_4"]
reg_losses_4 = sum(conv_var) + sum(p_4_var)
loss_op_4 = xnet_4 + (reg_constant * sum(reg_losses))
optimizer_4 = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_4 = optimizer_4.minimize(loss_op_4)
# personalize 5 
logits_5 = utils.personalized_model_5(tensor,keep_prob)
xnet_5 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_5, pos_weight=weight))
p_5_var = [v for v in tf.trainable_variables() if v.name == "p_5"]
reg_losses_5 = sum(conv_var) + sum(p_5_var)
loss_op_5 = xnet_5 + (reg_constant * reg_losses_5)
optimizer_5 = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_5 = optimizer_5.minimize(loss_op_5)
# personalize 6 
logits_6 = utils.personalized_model_6(tensor,keep_prob)
xnet_6 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_6, pos_weight=weight))
p_6_var = [v for v in tf.trainable_variables() if v.name == "p_6"]
reg_losses_6 = sum(conv_var) + sum(p_6_var)
loss_op_6 = xnet_6 + (reg_constant * sum(reg_losses))
optimizer_6 = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_6 = optimizer_6.minimize(loss_op_6)
# personalize 7 
logits_7 = utils.personalized_model_7(tensor,keep_prob)
xnet_7 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_7, pos_weight=weight))
p_7_var = [v for v in tf.trainable_variables() if v.name == "p_7"]
reg_losses_7 = sum(conv_var) + sum(p_7_var)
loss_op_7 = xnet_7 + (reg_constant * reg_losses_7)
optimizer_7 = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_7 = optimizer_7.minimize(loss_op_7)
# personalize 8 
logits_8 = utils.personalized_model_8(tensor,keep_prob)
xnet_8 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_7, pos_weight=weight))
p_8_var = [v for v in tf.trainable_variables() if v.name == "p_8"]
reg_losses_8 = sum(conv_var) + sum(p_8_var)
loss_op_8 = xnet_8 + (reg_constant * reg_losses_8)
optimizer_8 = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_8 = optimizer_8.minimize(loss_op_8)
# model saver initializer 
saver = tf.train.Saver()
# Start training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mean_train_loss = []
    mean_val_loss = []
    accuracy = []
    TP = []
    TN = []
    FP = []
    FN = []
    patience_cnt = 0
    check_loss = []
    for i in range(EPOCHS):
      prediction = []
      ground_truth = []
      train_loss_1 = []
      train_loss_2 = []
      train_loss_3 = []
      train_loss_4 = []
      train_loss_5 = []
      train_loss_6 = []
      train_loss_7 = []
      train_loss_8 = []
      val_loss= []
      # datasets
      samples_1 = train_samples[0]
      labels_1 = train_labels[0]
      samples_2 = train_samples[1]
      labels_2 = train_labels[1]
      samples_3 = train_samples[2]
      labels_3 = train_labels[2]
      samples_4 = train_samples[3]
      labels_4 = train_labels[3]
      samples_5 = train_samples[4]
      labels_5 = train_labels[4]
      samples_6 = train_samples[5]
      labels_6 = train_labels[5]
      samples_7 = train_samples[6]
      labels_7 = train_labels[6]
      samples_8 = train_samples[7]
      labels_8 = train_labels[7]
      # training 
      print('EPOCH #', i)
      all_samples = []
      all_labels = []
      samples_1, labels_1 = utils.get_shuffle_batch(samples_1,labels_1)
      samples_2, labels_2 = utils.get_shuffle_batch(samples_2,labels_2)
      samples_3, labels_3 = utils.get_shuffle_batch(samples_3,labels_3)
      samples_4, labels_4 = utils.get_shuffle_batch(samples_4,labels_4)
      samples_5, labels_5 = utils.get_shuffle_batch(samples_5,labels_5)
      samples_6, labels_6 = utils.get_shuffle_batch(samples_6,labels_6)
      samples_7, labels_7 = utils.get_shuffle_batch(samples_7,labels_7)
      samples_8, labels_8 = utils.get_shuffle_batch(samples_8,labels_8)
      # samples
      all_samples.append(samples_1)
      all_samples.append(samples_2)
      all_samples.append(samples_3)
      all_samples.append(samples_4)
      all_samples.append(samples_5)
      all_samples.append(samples_6)
      all_samples.append(samples_7)
      all_samples.append(samples_8)
      # labels
      all_labels.append(labels_1)
      all_labels.append(labels_2)
      all_labels.append(labels_3)
      all_labels.append(labels_4)
      all_labels.append(labels_5)
      all_labels.append(labels_6)
      all_labels.append(labels_7)
      all_labels.append(labels_8)
      # initialize annotation for training  
      select_annoation = 0
      for j in range(0,len(all_samples[0]),params.batch_size):
          # take dataset
          samples = all_samples[select_annoation]
          labels = all_labels[select_annoation]
          neg_count = len(list(filter(lambda x: (x < 1), labels)))
          pos_count = len(list(filter(lambda x: (x > 0), labels)))
          training_weight = neg_count/pos_count
          if j == int(len(samples)/params.batch_size)*params.batch_size:
            input_x = samples[j:]
            input_y = labels[j:]
            input_y = np.reshape(input_y,[len(input_y),1])
          else:
            input_x = samples[j:j+params.batch_size]
            input_y = labels[j:j+params.batch_size]
            input_y = np.reshape(input_y,[params.batch_size,1])
          # words id's 
          train_word_ids = utils.get_word_ids(input_x,vocabulary)
          # conv features 
          conv_features = sess.run([conv_output], feed_dict={ x: train_word_ids})
          conv_features = np.squeeze(conv_features)
          # train loss with respective to each annotator 
          if select_annoation == 0:
            model_output_1, loss_1,_ = sess.run([logits_1,loss_op_1,train_1], feed_dict={ tensor: conv_features, y: input_y, keep_prob:0.8, weight:training_weight})
            train_loss_1.append(loss_1)
          elif select_annoation == 1:
            model_output_2, loss_2,_ = sess.run([logits_2,loss_op_2,train_2], feed_dict={ tensor: conv_features, y: input_y, keep_prob:0.8, weight:training_weight})
            train_loss_2.append(loss_2)
          elif select_annoation == 2:
            model_output_3, loss_3,_ = sess.run([logits_3,loss_op_3,train_3], feed_dict={ tensor: conv_features, y: input_y, keep_prob:0.8, weight:training_weight})
            train_loss_3.append(loss_3)
          elif select_annoation == 3:
            model_output_4, loss_4,_ = sess.run([logits_4,loss_op_4,train_4], feed_dict={ tensor: conv_features, y: input_y, keep_prob:0.8, weight:training_weight})
            train_loss_4.append(loss_4)
          elif select_annoation == 4:
            model_output_5, loss_5,_ = sess.run([logits_5,loss_op_5,train_5], feed_dict={ tensor: conv_features, y: input_y, keep_prob:0.8, weight:training_weight})
            train_loss_5.append(loss_5)
          elif select_annoation == 5:
            model_output_6, loss_6,_ = sess.run([logits_6,loss_op_6,train_6], feed_dict={ tensor: conv_features, y: input_y, keep_prob:0.8, weight:training_weight})
            train_loss_6.append(loss_6)
          elif select_annoation == 6:
            model_output_7, loss_7,_ = sess.run([logits_7,loss_op_7,train_7], feed_dict={ tensor: conv_features, y: input_y, keep_prob:0.8, weight:training_weight})
            train_loss_7.append(loss_7)
          elif select_annoation == 7:
            model_output_8, loss_8,_ = sess.run([logits_8,loss_op_8,train_8], feed_dict={ tensor: conv_features, y: input_y, keep_prob:0.8, weight:training_weight})
            train_loss_8.append(loss_8)
          select_annoation += 1
          if select_annoation > 7:
            select_annoation = 0
      # save mean train loss 
      print('Training Loss_1',np.mean(train_loss_1))
      print('Training Loss_2',np.mean(train_loss_2))
      print('Training Loss_3',np.mean(train_loss_3))
      print('Training Loss_4',np.mean(train_loss_4))
      print('Training Loss_5',np.mean(train_loss_5))
      print('Training Loss_6',np.mean(train_loss_6))
      print('Training Loss_7',np.mean(train_loss_7))
      print('Training Loss_8',np.mean(train_loss_8))
      # validation 
      # select validation annotation from 0 to 6 where 0:anno_1, 1:anno_2, 2:anno_3, 3:anno_4, 4:anno_5, 5:anno_6, 6:anno_7, 7:avg_anno 
      annotation_no = 0 
      samples = val_samples[annotation_no]
      labels = val_labels[annotation_no]
      neg_count = len(list(filter(lambda x: (x < 1), labels)))
      pos_count = len(list(filter(lambda x: (x > 0), labels)))
      training_weight = neg_count/pos_count
      for k in range(0,len(samples),params.batch_size):
        if k == int(len(samples)/params.batch_size)*params.batch_size:
          input_x = samples[k:]
          input_y = labels[k:]
          ground_truth.append(input_y)
          input_y = np.reshape(input_y,[len(input_y),1])
        else:
          input_x = samples[k:k+params.batch_size]
          input_y = labels[k:k+params.batch_size]
          ground_truth.append(input_y)
          input_y = np.reshape(input_y,[params.batch_size,1])
        # words id's ###
        test_word_ids = utils.get_word_ids(input_x,vocabulary)
        conv_features = sess.run([conv_output], feed_dict={ x: test_word_ids})
        conv_features = np.squeeze(conv_features)
        # val loss with respective to each annotator
        if val_annotation_no == 0:
          model_output, loss = sess.run([logits_1,loss_op_1], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:training_weight})
          val_loss.append(loss)
          prediction.append(model_output)
        elif val_annotation_no == 1:
          model_output, loss = sess.run([logits_2,loss_op_2], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:training_weight})
          val_loss.append(loss)
          prediction.append(model_output)
        elif val_annotation_no == 2:
          model_output, loss = sess.run([logits_3,loss_op_3], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:training_weight})
          val_loss.append(loss)
          prediction.append(model_output)
        elif val_annotation_no == 3:
          model_output, loss = sess.run([logits_4,loss_op_4], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:training_weight})
          val_loss.append(loss)
          prediction.append(model_output)
        elif val_annotation_no == 4:
          model_output, loss = sess.run([logits_5,loss_op_5], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:training_weight})
          val_loss.append(loss)
          prediction.append(model_output)
        elif val_annotation_no == 5:
          model_output, loss = sess.run([logits_6,loss_op_6], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:training_weight})
          val_loss.append(loss)
          prediction.append(model_output)
        elif val_annotation_no == 6:
          model_output, loss = sess.run([logits_7,loss_op_7], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:training_weight})
          val_loss.append(loss)
          prediction.append(model_output)
        elif val_annotation_no == 7:
          model_output, loss = sess.run([logits_8,loss_op_8], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:training_weight})
          val_loss.append(loss)
          prediction.append(model_output)
      # save mean val loss 
      mean_val_loss.append(np.mean(val_loss))
      print('Validation Loss',np.mean(val_loss))
      check_loss.append(np.mean(val_loss))
      pred_vector = utils.get_vector(prediction)
      gt_vector = utils.get_vector(ground_truth)
      # binary 
      pred_binary = utils.get_binary(pred_vector)
      gt_binary = utils.get_binary(gt_vector)
      # early stopping
      patience = 10     
      min_delta = 0.01
      if i == 0:
        best_loss = check_loss[i]
      if i > 0 and check_loss[i] < best_loss - min_delta:
          best_loss = check_loss[i]
          epoch_no = i
          patience_cnt = 0
          save_path = ".../path/to/save/the/trained/model/CNN_Model"
          saver.save(sess, save_path)
      else:
          patience_cnt += 1
      if patience_cnt > patience:
          print("early stopping...")
          break

# testing
load_path = ".../path/to/save/the/trained/model/CNN_Model.meta"
saver = tf.train.import_meta_graph(load_path)
# Start testing
with tf.Session() as sess:
  saver.restore(sess, tf.train.latest_checkpoint(".../path/to/save/the/trained/model/"))
  print("Model restored.")
  ground_truth = []
  prediction = []
  samples = test_samples[annotation_no]
  labels = test_labels[annotation_no]
  neg_count = len(list(filter(lambda x: (x < 1), labels)))
  pos_count = len(list(filter(lambda x: (x > 0), labels)))
  test_weight = neg_count/pos_count
  for k in range(0,len(samples),params.batch_size):
    if k == int(len(samples)/params.batch_size)*params.batch_size:
      input_x = samples[k:]
      input_y = labels[k:]
      sentence.append(input_x)
      ground_truth.append(input_y)
      input_y = np.reshape(input_y,[len(input_y),1])
    else:
      input_x = samples[k:k+params.batch_size]
      input_y = labels[k:k+params.batch_size]
      sentence.append(input_x)
      ground_truth.append(input_y)
      input_y = np.reshape(input_y,[batch_size,1])
    # words id's 
    test_word_ids = utils.get_word_ids(input_x,vocabulary)
    conv_features = sess.run([conv_output], feed_dict={ x: test_word_ids})
    conv_features = np.squeeze(conv_features)
    if annotation_no == 0:
      model_output = sess.run([logits_1], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:test_weight})
      prediction.append(model_output)
    elif annotation_no == 1:
      model_output = sess.run([logits_2], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:test_weight})
      prediction.append(model_output)
    elif annotation_no == 2:
      model_output = sess.run([logits_3], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:test_weight})
      prediction.append(model_output)
    elif annotation_no == 3:
      model_output = sess.run([logits_4], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:test_weight})
      prediction.append(model_output)
    elif annotation_no == 4:
      model_output = sess.run([logits_5], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:test_weight})
      prediction.append(model_output)
    elif annotation_no == 5:
      model_output = sess.run([logits_6], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:test_weight})
      prediction.append(model_output)
    elif annotation_no == 6:
      model_output = sess.run([logits_7], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:test_weight})
      prediction.append(model_output)
    elif annotation_no == 7:
      model_output = sess.run([logits_8], feed_dict={ tensor: conv_features, y: input_y, keep_prob:1.0, weight:test_weight})
      prediction.append(model_output)
  pred_vector = utils.get_vector(prediction)
  gt_vector = utils.get_vector(ground_truth)
  # binary 
  pred_binary = utils.get_binary(pred_vector)
  gt_binary = utils.get_binary(gt_vector)
  print('Test Results:',utils.md_performance(gt_binary,pred_binary))
  
