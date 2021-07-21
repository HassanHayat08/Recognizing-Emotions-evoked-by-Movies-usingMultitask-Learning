# Recognizing Emotions evoked by Movies using Multitask Learning #
# Hassan Hayat, Carles Ventura, Agata Lapedriza #
# This script contains the training, validation, and test the SingleModality-SingleTask (ST-Text) model using cross-validation sets for all available viewers # 

# import libraries 
import tensorflow as tf
import numpy as np
import pandas as pd
from random import shuffle
import re, string
import csv
import random 
import os
import utils
import params

# all movies names #
movie_names = ["BMI","CHI","CRA","DEP","FNE","GLA","LOR"]
# all annotators #
all_annotators = ["experienced_1","experienced_2","experienced_3","experienced_4","experienced_5","experienced_6","experienced_7",'avg_experienced']
# create cross validation sets #
all_sets = utils.create_cross_validation_sets(movie_names)
# datasets for train,val and test with all necessary preprocessing 
# select annotator
select_annotator = 0  # select viewer no, select between 0 and 7
single_annotator = all_annotators[select_annotator]
# select cross-validation set no, select between 0 to 6
select_set = 0
# load data and divide in to train,val and test for all 7 sets of cross-validation 
# dimension of train,val and test are [7*no_of_samples(5 movies)], [7*no_of_samples(1 movie), [7*no_of_samples(1 movie)] respectively.
# Here 7 represents the no. of cross-validation sets for a single annotator   
all_sets_train_data, all_sets_val_data, all_sets_test_data = utils.create_train_val_test_datasets(all_sets,single_annotator)
# clean the data 
all_sets_train_samples, all_sets_train_labels = utils.data_cleaning(all_sets_train_data)
all_sets_val_samples, all_sets_val_labels = utils.data_cleaning(all_sets_val_data)
all_sets_test_samples, all_sets_test_labels = utils.data_cleaning(all_sets_test_data)
# get max length of sentence 
max_words = utils.max_length_of_sentence(all_sets_train_samples,all_sets_val_samples,all_sets_test_samples)
# data padding 
train_pad_samples = utils.data_padding(all_sets_train_samples,max_words)
val_pad_samples = utils.data_padding(all_sets_val_samples,max_words)
test_pad_samples = utils.data_padding(all_sets_test_samples,max_words)
# create vocabulary 
whole_text_path = '.../path/to/whole/data/whole_text.txt'
movie_subscripts,_ = utils.get_text_and_emotion(whole_text_path)
vocabulary = utils.get_vocab(movie_subscripts)
VOCAB_LEN = len(vocabulary)
# inputs and labels 
x = tf.placeholder(tf.int32, shape=([None,params.max_words]))
y = tf.placeholder(tf.float32, shape=([None,1]))
keep_prob = tf.placeholder(tf.float32, shape=())
weight = tf.placeholder(tf.float32, shape=())
# Store bias
biases = {
    'bc1': tf.get_variable('bc1',shape=[300], initializer=tf.zeros_initializer()),
    'bc2': tf.get_variable('bc2',shape=[300], initializer=tf.zeros_initializer()),
    'bc3': tf.get_variable('bc3',shape=[300], initializer=tf.zeros_initializer()),
    }
# construct model 
# create model
def conv_net(x, VOCAB_LEN, EMBED_SIZE, biases):
    # embedding layer 
    embed = tf.contrib.layers.embed_sequence(ids=x, vocab_size=VOCAB_LEN, embed_dim=EMBED_SIZE,
                  initializer=tf.truncated_normal_initializer(), regularizer=None,
                  trainable=True)
    net = tf.reshape(embed, shape=[-1, max_words, EMBED_SIZE, 1])
    # convolution layer
    conv1 = utils.conv2d(net, [2,EMBED_SIZE], biases['bc1'])
    # max pooling (down-sampling)
    conv1 = utils.maxpool2d(conv1, k=3)
    # convolution layer
    conv2 = utils.conv2d(conv1, [3,EMBED_SIZE], biases['bc2'])
    # max pooling (down-sampling)
    conv2 = utils.maxpool2d(conv2, k=3)
    # flatten layer
    flatten_layer = tf.contrib.layers.flatten(conv2)
    # fully connected #
    fc1 = tf.contrib.layers.fully_connected(flatten_layer ,1024, activation_fn=tf.nn.relu,
    weights_initializer=tf.truncated_normal_initializer(),
    biases_initializer=tf.zeros_initializer(),
    trainable=True)
    # fully connected
    fc2 = tf.contrib.layers.fully_connected(flatten_layer ,512, activation_fn=tf.nn.relu,
    weights_initializer=tf.truncated_normal_initializer(),
    biases_initializer=tf.zeros_initializer(),
    trainable=True)
    # dropout
    drop_out_2 = tf.nn.dropout(fc2, keep_prob)
    # fully connected
    fc3 = tf.contrib.layers.fully_connected(drop_out_2, 1, activation_fn=None,
    weights_initializer=tf.truncated_normal_initializer(),
    biases_initializer=tf.zeros_initializer(),
    trainable=True)
    return fc3
# construct model
logits = conv_net(x, VOCAB_LEN, params.EMBED_SIZE, biases)
# cross-entropy label loss
xnet = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits, pos_weight=weight))
# regularization 
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_constant = 0.01  # Choose an appropriate one.
loss_op = xnet + (reg_constant * sum(reg_losses))
optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
train = optimizer.minimize(loss_op)
# model saver initialization 
saver = tf.train.Saver()
sets = ["set_1","set_2","set_3","set_4","set_5","set_6","set_7"]
# Start training
with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      # take single set of cross-validation; consits of train,valand test data 
      # train 
      train_samples = train_pad_samples[select_set]
      train_labels = all_sets_train_labels[select_set]
      # val 
      val_samples = val_pad_samples[select_set]
      val_labels = all_sets_val_labels[select_set]
      # test 
      test_samples = train_pad_samples[select_set]
      test_labels = all_sets_test_labels[select_set]
      mean_train_loss = []
      mean_val_loss = []
      mean_test_loss = []
      accuracy = []
      TP = []
      TN = []
      FP = []
      FN = []
      patience_cnt = 0
      check_loss = []
      for epoch in range(EPOCHS):
        val_prediction = []
        test_prediction = []
        val_ground_truth = []
        test_ground_truth = []
        train_loss = []
        val_loss = []
        test_loss= []
        # training 
        print('EPOCH #', epoch)
        samples, labels = utils.get_shuffle_batch(train_samples,train_labels)
        neg_count = len(list(filter(lambda x: (x < 1), labels))) 
        pos_count = len(list(filter(lambda x: (x > 0), labels)))
        training_weight = neg_count/pos_count
        for k in range(0,len(samples),params.batch_size):
            if k == int(len(samples)/params.batch_size)*params.batch_size:
              input_x = samples[k:]
              input_y = labels[k:]
              input_y = np.reshape(input_y,[len(input_y),1])
            else:
              input_x = samples[k:k+params.batch_size]
              input_y = labels[k:k+params.batch_size]
              input_y = np.reshape(input_y,[params.batch_size,1])
            # words id's 
            train_word_ids = utils.get_word_ids(input_x,vocabulary)
            _, loss,_ = sess.run([logits,loss_op,train], feed_dict={ x: train_word_ids, y: input_y, keep_prob:0.8, weight:training_weight})
            train_loss.append(loss)
        # save mean train loss             
        mean_train_loss.append(np.mean(train_loss))
        print('Training Loss:',np.mean(train_loss))
        # validation 
        samples, labels = utils.get_shuffle_batch(val_samples,val_labels)
        neg_count = len(list(filter(lambda x: (x < 1), labels))) 
        pos_count = len(list(filter(lambda x: (x > 0), labels)))
        val_weight = neg_count/pos_count
        for l in range(0,len(samples),params.batch_size):
          if l == int(len(samples)/params.batch_size)*params.batch_size:
            input_x = samples[l:]
            input_y = labels[l:]
            val_ground_truth.append(input_y)
            input_y = np.reshape(input_y,[len(input_y),1])
          else:
            input_x = samples[l:l+params.batch_size]
            input_y = labels[l:l+params.batch_size]
            val_ground_truth.append(input_y)
            input_y = np.reshape(input_y,[params.batch_size,1])
          # words id's 
          val_word_ids = utils.get_word_ids(input_x,vocabulary)
          val_output, loss = sess.run([logits,loss_op], feed_dict={ x: val_word_ids, y: input_y, keep_prob:1.0, weight:val_weight})
          val_loss.append(loss)
          val_prediction.append(val_output)
        # save mean val loss             
        mean_val_loss.append(np.mean(val_loss))
        print('Validation Loss:',np.mean(val_loss))
        check_loss.append(np.mean(val_loss))
        val_pred_vector = utils.get_vector(val_prediction)
        val_gt_vector = utils.get_vector(val_ground_truth)
        # binary 
        val_pred_binary = utils.get_binary(val_pred_vector)
        val_gt_binary = utils.get_binary(val_gt_vector)
        # early stopping
        patience = 20
        min_delta = 0.01
        if epoch == 0:
          best_loss = check_loss[epoch]
        if epoch > 0 and check_loss[epoch] < best_loss - min_delta:
            best_loss = check_loss[epoch]
            patience_cnt = 0
            save_path = "./save_models/" + sets[select_set] + "/" + "model_no" + str(select_set)
            saver.save(sess, save_path)
        else:
            patience_cnt += 1
        if patience_cnt > patience:
            print("early stopping...")
            break
           
# Testing
# load saved model
load_path = ".../save_models/model_no" + str(select_set) +  "/" + "cnn_model.meta"
saver = tf.train.import_meta_graph(load_path)
# Start training
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(".../save_models/" + select_set + "/"))
    print("Model restored.")
    test_samples = test_pad_samples[select_set]
    test_labels = all_sets_test_labels[select_set]
    test_ground_truth = []
    test_prediction = []
    test_sentences = []
    # get only positive and negative examples 
    neg_count = len(list(filter(lambda x: (x < 1), test_labels)))
    pos_count = len(list(filter(lambda x: (x > 0), test_labels)))
    pos_samples, pos_labels, neg_samples, neg_labels = separate_neg_pos_ex(test_samples,test_labels)
    samples, labels = get_shuffle_batch(pos_samples, pos_labels)
    test_weight = neg_count/pos_count
    for l in range(0,len(samples),params.batch_size):
      if l == int(len(samples)/params.batch_size)*params.batch_size:
        input_x = samples[l:]
        input_y = labels[l:]
        test_sentences.append(input_x)
        test_ground_truth.append(input_y)
        input_y = np.reshape(input_y,[len(input_y),1])
      else:
        input_x = samples[l:l+params.batch_size]
        input_y = labels[l:l+params.batch_size]
        test_sentences.append(input_x)
        test_ground_truth.append(input_y)
        input_y = np.reshape(input_y,[params.batch_size,1])
      # words id's 
      test_word_ids = get_word_ids(input_x,vocabulary)
      test_output = sess.run([logits], feed_dict={ x: test_word_ids, y: input_y, keep_prob:1.0, weight:test_weight})
      test_prediction.append(test_output)
    test_pred_vector = get_vector(test_prediction)
    test_gt_vector = get_vector(test_ground_truth)
    # get binary 
    test_pred_binary = get_binary(test_pred_vector)
    test_gt_binary = get_binary(test_gt_vector)
    print('Test Results:',utils.md_performance(test_gt_binary,test_pred_binary))
           

