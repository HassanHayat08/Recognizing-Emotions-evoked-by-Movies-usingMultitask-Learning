# Multi-Model Multi-Task Learning for Emotion Recognition in Movies #
# Multitask learning using concatenation of the modalities #
# # This script contains the training, validation, and test the Multimodel-MultiTask (MT) using cross-validation sets for all availabel viewers #

# import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from random import shuffle
import csv
import os
import utils
import random
import params
# All movies names #
movie_names = ["BMI","CHI","CRA","DEP","FNE","GLA","LOR"]
# all annotators #
all_viewers = ["viewer_1","viewer_2","viewer_3","viewer_4","viewer_5","viewer_6","viewer_7","avg_all_viewers"]
# create cross validation sets #
all_sets = utils.create_cross_validation_sets(movie_names)
# datasets for train,val and test with all necessary preprocessing #
select_viewer = 0  # select viewer number between 0 and 7.
select_set_no = 0  # select cross-validation set no between 0 and 6
# load data and divide in to train,val and test for single set of cross-validation
visual_train, text_train, train_label, visual_val, text_val, val_label, visual_test, text_test, test_label = utils.create_train_val_test_datasets_mt_joint(all_sets,select_set_no,all_viewers,select_viewer)
# text preprocessing
# data cleaning
text_train = utils.data_cleaning(text_train)
text_val = utils.data_cleaning(text_val)
text_test = utils.data_cleaning(text_test)
# data padding
text_train = utils.data_padding(text_train,18)
text_val = utils.data_padding(text_val,18)
text_test = utils.data_padding(text_test,18)
# vocab
path = '.../path/to/the/whole/text/data/for/creating/vocabulary/whole_text.txt'
movie_subscripts,_ = utils.get_text_and_emotion(path)
vocabulary = utils.get_vocab(movie_subscripts)
VOCAB_LEN = len(vocabulary)
# inputs and labels 
x_conv = tf.placeholder(tf.int32, shape=([None,params.max_words]))
x_cmb = tf.placeholder(tf.float32, shape=([None,2224]))
x_per = tf.placeholder(tf.float32, shape=([None,1024]))
y = tf.placeholder(tf.float32, shape=([None,1]))
keep_prob = tf.placeholder(tf.float32, shape=())
weight = tf.placeholder(tf.float32, shape=())
reg_constant = 0.2  # Choose an appropriate one.
# store bias
biases = {
    'bc1': tf.get_variable('bc1',shape=[300], initializer=tf.zeros_initializer()),                                                            
    'bc2': tf.get_variable('bc2',shape=[300], initializer=tf.zeros_initializer()),                                                            
    'bc3': tf.get_variable('bc3',shape=[300], initializer=tf.zeros_initializer()),
    }
# Create convolutional model
def conv_net(x, VOCAB_LEN, EMBED_SIZE, biases):
    # embedding layer 
    embed = tf.contrib.layers.embed_sequence(ids=x, vocab_size=VOCAB_LEN, embed_dim=EMBED_SIZE,                                                 
                  initializer=tf.truncated_normal_initializer(), regularizer=None,
                  trainable=True)
    net = tf.reshape(embed, shape=[-1, 18, EMBED_SIZE, 1])
    # convolution layer
    conv1 = utils.conv2d(net, [2,EMBED_SIZE], biases['bc1'])
    # max pooling (down-sampling)
    conv1 = utils.maxpool2d(conv1, k=3)
    # convolution layer
    conv2 = utils.conv2d(conv1, [3,EMBED_SIZE], biases['bc2'])
    # max pooling (down-sampling)
    conv2 = utils.maxpool2d(conv2, k=3)
    # flatten
    flatten_layer = tf.contrib.layers.flatten(conv2)
    return flatten_layer

# convolutional features for text
conv_features = conv_net(x_conv, VOCAB_LEN, params.EMBED_SIZE, biases)
# shared fully-connected layers
cmb_fc_output = utils.cmb_fc(x_cmb)
# graph loss 
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
cmb_fc = [v for v in tf.trainable_variables() if v.name == "cmb_fc"]
# personalize 1 
logits_1 = utils.personalized_model_1(x_per,keep_prob)
xnet_1 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_1, pos_weight=weight))
p_1_var = [v for v in tf.trainable_variables() if v.name == "p_1"]
reg_losses_1 = sum(cmb_fc) + sum(p_1_var)
loss_op_1 = xnet_1 + (reg_constant * (reg_losses_1))
optimizer_1 = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
train_1 = optimizer_1.minimize(loss_op_1)
# personalize 2 
logits_2 = utils.personalized_model_2(x_per,keep_prob)
xnet_2 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_2, pos_weight=weight))
p_2_var = [v for v in tf.trainable_variables() if v.name == "p_2"]
reg_losses_2 = sum(cmb_fc) + sum(p_2_var)
loss_op_2 = xnet_2 + (reg_constant * (reg_losses_2))
optimizer_2 = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
train_2 = optimizer_2.minimize(loss_op_2)
# personalize 3 
logits_3 = utils.personalized_model_3(x_per,keep_prob)
xnet_3 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_3, pos_weight=weight))
p_3_var = [v for v in tf.trainable_variables() if v.name == "p_3"]
reg_losses_3 = sum(cmb_fc) + sum(p_3_var)
loss_op_3 = xnet_3 + (reg_constant * (reg_losses_3))
optimizer_3 = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
train_3 = optimizer_3.minimize(xnet_3)
# personalize 4 
logits_4 = utils.personalized_model_4(x_per,keep_prob)
xnet_4 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_4, pos_weight=weight))                             
p_4_var = [v for v in tf.trainable_variables() if v.name == "p_4"]
reg_losses_4 = sum(cmb_fc) + sum(p_4_var)
loss_op_4 = xnet_4 + (reg_constant * (reg_losses_4))
optimizer_4 = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
train_4 = optimizer_4.minimize(loss_op_4)
# personalize 5 
logits_5 = utils.personalized_model_5(x_per,keep_prob)
xnet_5 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_5, pos_weight=weight))
p_5_var = [v for v in tf.trainable_variables() if v.name == "p_5"]
reg_losses_5 = sum(cmb_fc) + sum(p_5_var)
loss_op_5 = xnet_5 + (reg_constant * (reg_losses_5))
optimizer_5 = tf.train.GradientDescentOptimizer(learning_rate=params.learning_rate)
train_5 = optimizer_5.minimize(loss_op_5)
# personalize 6 
logits_6 = utils.personalized_model_6(x_per,keep_prob)
xnet_6 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_6, pos_weight=weight))
p_6_var = [v for v in tf.trainable_variables() if v.name == "p_6"]
reg_losses_6 = sum(cmb_fc) + sum(p_6_var)
loss_op_6 = xnet_6 + (reg_constant * (reg_losses_6))
optimizer_6 = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
train_6 = optimizer_6.minimize(loss_op_6)
# personalize 7 
logits_7 = utils.personalized_model_7(x_per,keep_prob)
xnet_7 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_7, pos_weight=weight))
p_7_var = [v for v in tf.trainable_variables() if v.name == "p_7"]
reg_losses_7 = sum(cmb_fc) + sum(p_7_var)
loss_op_7 = xnet_7 + (reg_constant * (reg_losses_7))
optimizer_7 = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
train_7 = optimizer_7.minimize(loss_op_7)
# personalize 8
logits_8 = utils.personalized_model_8(x_per,keep_prob)
xnet_8 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits_8, pos_weight=weight))
p_8_var = [v for v in tf.trainable_variables() if v.name == "p_8"]
reg_losses_8 = sum(cmb_fc) + sum(p_8_var)
loss_op_8 = xnet_8 + (reg_constant * (reg_losses_8))
optimizer_8 = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
train_8 = optimizer_8.minimize(loss_op_8)
# model saver initializer #
saver = tf.train.Saver()
# start session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    check_loss = []
    patience_cnt = 0
    for epoch in range(params.EPOCHS):
        print('EPOCH #', epoch)
        # Training step
        prediction = []
        ground_truth = []
        all_labels = []
        train_loss_1 = []
        train_loss_2 = []
        train_loss_3 = []
        train_loss_4 = []
        train_loss_5 = []
        train_loss_6 = []
        train_loss_7 = []
        train_loss_8 = []
        val_loss= []
        train_label = np.asarray(train_label)
        train_label_1 = train_label[:,0]
        train_label_2 = train_label[:,1]
        train_label_3 = train_label[:,2]
        train_label_4 = train_label[:,3]
        train_label_5 = train_label[:,4]
        train_label_6 = train_label[:,5]
        train_label_7 = train_label[:,6]
        train_label_8 = train_label[:,7]
        all_labels.append(train_label_1)
        all_labels.append(train_label_2)
        all_labels.append(train_label_3)
        all_labels.append(train_label_4)
        all_labels.append(train_label_5)
        all_labels.append(train_label_6)
        all_labels.append(train_label_7)
        all_labels.append(train_label_8)
        all_labels = np.asarray(all_labels)
        idx_list = [i for i in range(len(train_label_1))]
        idx_list = random.sample(idx_list, len(idx_list))
        batch_info = []
        for i in range(0,len(idx_list),params.batch_size):
            batch_index = idx_list[i:i+params.batch_size]
            batch_info.append(batch_index)
        # get batch index info
        # initialize the viewer no
        viewer = 0
        for batch_no in range(len(batch_info)):
            single_batch = batch_info[batch_no]
            x_visual = []
            x_text = []
            input_y = []
            # select viewer
            labels = all_labels[viewer,:]
            neg_count = len(list(filter(lambda x: (x < 1), labels)))
            pos_count = len(list(filter(lambda x: (x > 0), labels)))
            # calculate pos weight
            training_weight = neg_count/pos_count
            for idx in range(len(single_batch)):
                x_visual.append(visual_train[single_batch[idx]])
                x_text.append(text_train[single_batch[idx]])
                input_y.append(labels[single_batch[idx]])
            input_y = np.reshape(input_y, (len(input_y),1))
            # words ID's 
            word_ids = utils.get_word_ids(x_text,vocabulary)
            # text features
            text_features = sess.run([conv_features], feed_dict={x_conv:word_ids})                                                          
            text_features = np.squeeze(text_features)
            # combine modalities
            cmb_features = np.concatenate((text_features, x_visual), axis=1)   
            # shared fully-connected layers  
            cmb_fc = sess.run([cmb_fc_output],feed_dict={x_cmb:cmb_features, keep_prob:0.6})
            cmb_fc = np.squeeze(cmb_fc)
            # training personalized models
            if viewer == 0:
                model_output_1, loss_1,_ = sess.run([logits_1,xnet_1,train_1], 
                        feed_dict={x_per:cmb_fc, y:input_y, keep_prob:0.6,weight:training_weight})
                train_loss_1.append(loss_1)
            elif viewer == 1:
                model_output_2, loss_2,_ = sess.run([logits_2,xnet_2,train_2], 
                        feed_dict={x_per:cmb_fc, y:input_y, keep_prob:0.6, weight:training_weight})
                train_loss_2.append(loss_2)
            elif viewer == 2:
                model_output_3, loss_3,_ = sess.run([logits_3,xnet_3,train_3], 
                        feed_dict={x_per:cmb_fc, y:input_y, keep_prob:0.6, weight:training_weight})
                train_loss_3.append(loss_3)
            elif viewer == 3:
                model_output_4, loss_4,_ = sess.run([logits_4,xnet_4,train_4], 
                        feed_dict={x_per:cmb_fc, y:input_y, keep_prob:0.6, weight:training_weight})
                train_loss_4.append(loss_4)
            elif viewer == 4:
                model_output_5, loss_5,_ = sess.run([logits_5,xnet_5,train_5], 
                        feed_dict={x_per:cmb_fc, y:input_y, keep_prob:0.6, weight:training_weight})
                train_loss_5.append(loss_5)
            elif viewer == 5:
                model_output_6, loss_6,_ = sess.run([logits_6,xnet_6,train_6], 
                        feed_dict={x_per:cmb_fc, y:input_y, keep_prob:0.6, weight:training_weight})
                train_loss_6.append(loss_6)
            elif viewer == 6:
                model_output_7, loss_7,_ = sess.run([logits_7,xnet_7,train_7], 
                        feed_dict={x_per:cmb_fc, y:input_y, keep_prob:0.6, weight:training_weight})
                train_loss_7.append(loss_7)
            elif viewer == 7:
                model_output_8, loss_8,_ = sess.run([logits_8,xnet_8,train_8],
                        feed_dict={x_per:cmb_fc, y:input_y, keep_prob:0.6, weight:training_weight})
                train_loss_8.append(loss_8)
            viewer += 1
            if viewer > 7:
                viewer = 0
        print('Training Loss_1',np.mean(train_loss_1))
        print('Training Loss_2',np.mean(train_loss_2))
        print('Training Loss_3',np.mean(train_loss_3))
        print('Training Loss_4',np.mean(train_loss_4))
        print('Training Loss_5',np.mean(train_loss_5))
        print('Training Loss_6',np.mean(train_loss_6))
        print('Training Loss_7',np.mean(train_loss_7))
        print('Training Loss_8',np.mean(train_loss_8))
        # validation
        neg_count = len(list(filter(lambda x: (x < 1), val_label)))
        pos_count = len(list(filter(lambda x: (x > 0), val_label)))
        training_weight = neg_count/pos_count
        for k in range(0,len(val_label),params.batch_size):
            if k == int(len(val_label)/params.batch_size)*params.batch_size:
                x_visual = visual_val[k:]
                x_text = text_val[k:]
                input_y = val_label[k:]
                ground_truth.append(input_y)
                input_y = np.reshape(input_y,[len(input_y),1])
            else:
                x_visual = visual_val[k:k+params.batch_size]
                x_text = text_val[k:k+params.batch_size]
                input_y = val_label[k:k+params.batch_size]
                ground_truth.append(input_y)
                input_y = np.reshape(input_y,[len(input_y),1])
            # words ID's 
            word_ids = utils.get_word_ids(x_text,vocabulary)
            # text features
            text_features = sess.run([conv_features], feed_dict={x_conv:word_ids})
            text_features = np.squeeze(text_features)
            # combine modalities
            cmb_features = np.concatenate((text_features, x_visual), axis=1)
            # shared fully-connected layers 
            cmb_fc = sess.run([cmb_fc_output],feed_dict={x_cmb:cmb_features, keep_prob:1.0})
            cmb_fc = np.squeeze(cmb_fc)
            # personalized 
            if select_viewer == 0:
                model_output, loss = sess.run([logits_1,xnet_1], 
                        feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
                val_loss.append(loss)
                prediction.append(model_output)
            elif select_viewer == 1:
                model_output, loss = sess.run([logits_2,xnet_2], 
                        feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
                val_loss.append(loss)
                prediction.append(model_output)
            elif select_viewer == 2:
                model_output, loss = sess.run([logits_3,xnet_3], 
                        feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
                val_loss.append(loss)
                prediction.append(model_output)
            elif select_viewer == 3:
                model_output, loss = sess.run([logits_4,xnet_4], 
                        feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
                val_loss.append(loss)
                prediction.append(model_output)
            elif select_viewer == 4:
                model_output, loss = sess.run([logits_5,xnet_5], 
                        feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
                val_loss.append(loss)
                prediction.append(model_output)
            elif select_viewer == 5:
                model_output, loss = sess.run([logits_6,xnet_6], 
                        feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
                val_loss.append(loss)
                prediction.append(model_output)
            elif select_viewer == 6:
                model_output, loss = sess.run([logits_7,xnet_7], 
                        feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
                val_loss.append(loss)
                prediction.append(model_output)
            elif select_viewer == 7:
                model_output, loss = sess.run([logits_8,xnet_8],
                        feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
                val_loss.append(loss)
                prediction.append(model_output)
        check_loss.append(np.mean(val_loss))
        print('Validation Loss',np.mean(val_loss))
        pred_vector = utils.get_vector(prediction)
        gt_vector = utils.get_vector(ground_truth)
        # binary #
        pred_binary = utils.get_binary(pred_vector)
        gt_binary = utils.get_binary(gt_vector)
        # early stopping
        patience = 25
        if epoch == 0:
            best_loss = check_loss[epoch]
        if epoch > 0 and check_loss[epoch] < best_loss:
            best_loss = check_loss[epoch]
            epoch_no = epoch
            patience_cnt = 0
            save_path = "./save/" + "cnn_model"
            saver.save(sess, save_path)
        else:
            patience_cnt += 1
        if patience_cnt > patience:
            print("early stopping...")
            break
# Testing
# load model
load_path = "./save/" + "cnn_model.meta"
saver = tf.train.import_meta_graph(load_path)
# Start training
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint("./save/"))
    print("Model restored.")
    prediction = []
    ground_truth = []
    # testing
    neg_count = len(list(filter(lambda x: (x < 1), test_label)))
    pos_count = len(list(filter(lambda x: (x > 0), test_label)))
    training_weight = pos_count/neg_count
    for k in range(0,len(test_label),params.batch_size):
        if k == int(len(test_label)/params.batch_size)*params.batch_size:
            x_visual = visual_test[k:]
            x_text = text_test[k:]
            input_y = test_label[k:]
            ground_truth.append(input_y)
            input_y = np.reshape(input_y,[len(input_y),1])
        else:
            x_visual = visual_test[k:k+params.batch_size]
            x_text = text_test[k:k+params.batch_size]
            input_y = test_label[k:k+params.batch_size]
            ground_truth.append(input_y)
            input_y = np.reshape(input_y,[len(input_y),1])
        # words ID's 
        word_ids = utils.get_word_ids(x_text,vocabulary)
        # text features
        text_features = sess.run([conv_features], feed_dict={x_conv:word_ids})
        text_features = np.squeeze(text_features)
        # combine modalities
        cmb_features = np.concatenate((text_features, x_visual), axis=1)
        # Fc block for combine modalities before feeding to personalized 
        cmb_fc = sess.run([cmb_fc_output],feed_dict={x_cmb:cmb_features, keep_prob:0.6})
        cmb_fc = np.squeeze(cmb_fc)
        # personalized testing
        if select_viewer == 0:
            model_output = sess.run([logits_1], 
                    feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
            prediction.append(model_output)
        elif select_viewer == 1:
            model_output = sess.run([logits_2], 
                    feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
            prediction.append(model_output)
        elif select_viewer == 2:
            model_output = sess.run([logits_3], 
                    feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
            prediction.append(model_output)
        elif select_viewer == 3:
            model_output = sess.run([logits_4], 
                    feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
            prediction.append(model_output)
        elif select_viewer == 4:
            model_output = sess.run([logits_5], 
                    feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
            prediction.append(model_output)
        elif select_viewer == 5:
            model_output = sess.run([logits_6], 
                    feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
            prediction.append(model_output)
        elif select_viewer == 6:
            model_output = sess.run([logits_7], 
                    feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
            prediction.append(model_output)
        elif select_viewer == 7:
            model_output = sess.run([logits_8],
                    feed_dict={x_per:cmb_fc, y: input_y, keep_prob:1.0, weight:training_weight})
            prediction.append(model_output)
    prediction = np.squeeze(prediction)
    pred_vector = utils.get_vector(prediction)
    gt_vector = utils.get_vector(ground_truth)
    # binary #
    pred_binary = utils.get_binary(pred_vector)
    gt_binary = utils.get_binary(gt_vector)
    print('Test Results:', utils.md_performance(gt_binary,pred_binary))

