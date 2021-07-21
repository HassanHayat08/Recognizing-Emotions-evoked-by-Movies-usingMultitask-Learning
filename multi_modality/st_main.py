# Recognizing Emotions evoked by Movies using Multitask Learning #
# Hassan Hayat, Carles Ventura, Agata Lapedriza #
# This script contains the training, validation, and test the Multimodel-SingleTask (ST) model using cross-validation sets for all availabel viewers # 

# import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from random import shuffle
import csv
import os
import utils
from scipy import stats
from sklearn.preprocessing import normalize
import params

# all movies names 
movie_names = ["BMI","CHI","CRA","DEP","FNE","GLA","LOR"]
# all annotators 
all_viewers = ["viewer_1","viewer_2","viewer_3","viewer_4","viewer_5","viewer_6","viewer_7","avg_all_viewers"]
# create cross validation sets 
all_sets = utils.create_cross_validation_sets(movie_names)
# select viewer 
select_viewer = 0  # select viewer no, select between 0 and 7
single_viewer = all_viewers[select_viewer]
select_set_no = 0 # select cross-validation set no, select between 0 and 6
visual_train, text_train, audio_train, train_label, visual_val, text_val, audio_val, val_label, visual_test, text_test, audio_test, test_label = utils.create_train_val_test_datasets_st_joint(all_sets,select_set_no,single_viewer)
# data cleaning
text_train = utils.data_cleaning(text_train)
text_val = utils.data_cleaning(text_val)
text_test = utils.data_cleaning(text_test)
# data padding 
text_train = utils.data_padding(text_train,params.max_words)
text_val = utils.data_padding(text_val,params.max_words)
text_test = utils.data_padding(text_test,params.max_words)
# vocab
path = '.../path/to/the/whole/text/data/for/creating/vocabulary/whole_text.txt'
movie_subscripts,_ = utils.get_text_and_emotion(path)
vocabulary = utils.get_vocab(movie_subscripts)
VOCAB_LEN = len(vocabulary)
# inputs and labels 
x = tf.placeholder(tf.int32, shape=([None,params.max_words]))
y = tf.placeholder(tf.float32, shape=([None,1]))
tensor = tf.placeholder(tf.float32, shape=([None,params.cmb_tensor_size]))
keep_prob = tf.placeholder(tf.float32, shape=())
weight = tf.placeholder(tf.float32, shape=())
# store bias
biases = {
    'bc1': tf.get_variable('bc1',shape=[300], initializer=tf.zeros_initializer()),
    'bc2': tf.get_variable('bc2',shape=[300], initializer=tf.zeros_initializer()),
    'bc3': tf.get_variable('bc3',shape=[300], initializer=tf.zeros_initializer()),
    }
# Create model
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
    # flatten layer
    flatten_layer = tf.contrib.layers.flatten(conv2)
    return flatten_layer

def MLP(tensor,keep_prob):
    # fully connected 
    fc1 = tf.contrib.layers.fully_connected(tensor, 1024, activation_fn=tf.nn.relu,
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
    dp2 = tf.nn.dropout(fc3,keep_prob)
    # output
    fc4 = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
    weights_initializer=tf.random_normal_initializer(seed=10),
    biases_initializer=tf.zeros_initializer(),
    trainable=True)
    return fc4

conv_features = conv_net(x, VOCAB_LEN, params.EMBED_SIZE, biases)
logits = MLP(tensor,keep_prob)
# weighted cross-entropy label loss.
xnet = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=logits, pos_weight=weight))
# regularization 
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_constant = 0.4  # Choose an appropriate one.
loss_op = xnet + (reg_constant * sum(reg_losses))
optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
train = optimizer.minimize(loss_op)
# model saver initialization #
saver = tf.train.Saver()
# start session
with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      patience_cnt = 0
      check_loss = []
      for epoch in range(params.EPOCHS):
        train_loss = []
        val_loss = []
        predictions = []
        targets = []
        # training 
        print('EPOCH #', epoch)
        text_samples, visual_samples, audio_samples, labels = utils.get_shuffle_batch(text_train,visual_train,audio_train, train_label)
        neg_count = len(list(filter(lambda x: (x < 1), labels)))
        pos_count = len(list(filter(lambda x: (x > 0), labels)))
        training_weight = neg_count/pos_count
        for k in range(0,len(text_samples),params.batch_size):
            if k == int(len(text_samples)/params.batch_size)*params.batch_size:
              text_input = text_samples[k:]
              visual_features = np.asarray(visual_samples[k:])
              audio_features = np.asarray(audio_samples[k:])
              input_y = labels[k:]
              input_y = np.reshape(input_y,[len(input_y),1])
            else:
              text_input = text_samples[k:k+params.batch_size]
              visual_features = np.asarray(visual_samples[k:k+params.batch_size])
              audio_features = np.asarray(audio_samples[k:k+params.batch_size])
              input_y = labels[k:k+params.batch_size]
              input_y = np.reshape(input_y,[params.batch_size,1])
            # words ID's 
            word_ids = utils.get_word_ids(text_input,vocabulary)
            # text features
            text_features = sess.run([conv_features], feed_dict={ x:word_ids})
            text_features = np.squeeze(text_features)
            # combine modalities
            cmb_txt_vis_features = np.concatenate((text_features, visual_features), axis=1)
            cmb_features = np.concatenate((cmb_txt_vis_features, audio_features), axis=1)
            cmb_features = normalize(cmb_features)
            loss,_ = sess.run([xnet,train], feed_dict={ tensor:cmb_features, y:input_y, keep_prob:0.6, weight:training_weight})
            train_loss.append(loss)
        print('Training Loss:', np.mean(train_loss))
        # validation 
        text_samples,visual_samples,audio_samples, labels = utils.get_shuffle_batch(text_val,visual_val,audio_val, val_label)
        neg_count = len(list(filter(lambda x: (x < 1), labels)))
        pos_count = len(list(filter(lambda x: (x > 0), labels)))
        val_weight = neg_count/pos_count
        for k in range(0,len(text_samples),params.batch_size):
            if k == int(len(text_samples)/params.batch_size)*params.batch_size:
              text_input = text_samples[k:]
              visual_features = np.asarray(visual_samples[k:])
              audio_features = np.asarray(audio_samples[k:])
              input_y = labels[k:]
              targets.append(input_y)
              input_y = np.reshape(input_y,[len(input_y),1])
            else:
              text_input = text_samples[k:k+params.batch_size]
              visual_features = np.asarray(visual_samples[k:k+params.batch_size])
              audio_features = np.asarray(audio_samples[k:k+params.batch_size])
              input_y = labels[k:k+params.batch_size]
              targets.append(input_y)
              input_y = np.reshape(input_y,[params.batch_size,1])
            # words ID's 
            word_ids = utils.get_word_ids(text_input,vocabulary)
            # text features
            text_features = sess.run([conv_features], feed_dict={ x:word_ids})
            text_features = np.squeeze(text_features)
            # combine modalities
            cmb_txt_vis_features = np.concatenate((text_features, visual_features), axis=1)
            cmb_features = np.concatenate((cmb_txt_vis_features, audio_features), axis=1)
            cmb_features = normalize(cmb_features)
            output,loss = sess.run([logits,xnet], feed_dict={ tensor:cmb_features, y:input_y, keep_prob:1.0, weight:val_weight})
            val_loss.append(loss)
            predictions.append(output)
        print('Validation Loss:',np.mean(val_loss))
        check_loss.append(np.mean(val_loss))
        pred_vector = utils.get_vector(predictions)
        gt_vector = utils.get_vector(targets)
        # get binary 
        pred_binary = utils.get_binary(pred_vector)
        gt_binary = utils.get_binary(gt_vector)
        # early stopping
        patience = 20
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
# Start session
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint("./save/"))
    print("Model restored.")
    test_gt = []
    test_md = []
    text_samples,visual_samples,audio_samples, labels = utils.get_shuffle_batch(text_test,visual_test,audio_test,test_label)
    neg_count = len(list(filter(lambda x: (x < 1), labels)))
    pos_count = len(list(filter(lambda x: (x > 0), labels)))
    test_weight = neg_count/pos_count
    for k in range(0,len(text_samples),params.batch_size):
        if k == int(len(text_samples)/params.batch_size)*params.batch_size:
            text_input = text_samples[k:]
            visual_features = np.asarray(visual_samples[k:])
            audio_features = np.asarray(audio_samples[k:])
            input_y = labels[k:]
            test_gt.append(input_y)
            input_y = np.reshape(input_y,[len(input_y),1])
        else:
            text_input = text_samples[k:k+params.batch_size]
            visual_features = np.asarray(visual_samples[k:k+params.batch_size])
            audio_features = np.asarray(audio_samples[k:k+params.batch_size])
            input_y = labels[k:k+params.batch_size]
            test_gt.append(input_y)
            input_y = np.reshape(input_y,[params.batch_size,1])
        # words ID's 
        word_ids = utils.get_word_ids(text_input,vocabulary)
        # text features
        text_features = sess.run([conv_features], feed_dict={ x:word_ids})
        text_features = np.squeeze(text_features)
        # combine modalities
        cmb_txt_vis_features = np.concatenate((text_features, visual_features), axis=1)
        cmb_features = np.concatenate((cmb_txt_vis_features, audio_features), axis=1)
        cmb_features = normalize(cmb_features)
        # cmb feature training
        output = sess.run([logits], feed_dict={tensor:cmb_features, y:input_y, keep_prob:1.0, weight:test_weight})
        output = np.squeeze(output)
        test_md.append(output)
    
    pred_vector = utils.get_vector(test_md)
    gt_vector = utils.get_vector(test_gt)
    # get binary 
    pred_binary = utils.get_binary(pred_vector)
    gt_binary = utils.get_binary(gt_vector)
    print('Test Results:',utils.md_performance(gt_binary,pred_binary))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
