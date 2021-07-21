# Recognizing Emotions evoked by Movies using Multitask Learning #
# Hassan Hayat, Carles Ventura, Agata Lapedriza #
# This script contains supporting functions #

# import libraries
import numpy as np
import tensorflow as tf
import pandas as pd
from random import shuffle
import csv
import random 
import os
import glob
from scipy import stats
from sklearn.preprocessing import StandardScaler

# function to create cross validation sets 
def create_cross_validation_sets(movies_set):
    sets = ["set_1","set_2","set_3","set_4","set_5","set_6","set_7"]
    cross_validation_sets = {x: {} for x in sets}
    for i in range(7):
        train_mv = []
        val_mv = []
        test_mv = []
        val_mv_no = i+1
        if val_mv_no == 7:
            val_mv_no = 1
        for j in range(7):
            if j == i:
                test_mv.append(movies_set[i])
            elif j == val_mv_no:
                val_mv.append(movies_set[val_mv_no])
            elif j != i and j != val_mv_no:
                mv_name = movies_set[j]
                train_mv.append(mv_name)
        set_no = list(cross_validation_sets.values())[i]
        set_no["train_mv"] = train_mv
        set_no["val_mv"] = val_mv
        set_no["test_mv"] = test_mv
    return cross_validation_sets

# function to create train, val and test datasets for all cross_validation sets 
def create_train_val_test_datasets_st(sets,select_set,viewer):
    set_no = list(sets.values())[select_set]
    train_mv_list = set_no["train_mv"]
    val_mv_list = set_no["val_mv"]
    test_mv_list = set_no["test_mv"]
    cross_validation_list = []
    cross_validation_list.append(train_mv_list)
    cross_validation_list.append(val_mv_list)
    cross_validation_list.append(test_mv_list)
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []
    # take train , val and test data for single cross_validation set 
    # get visual features
    for j in range(len(cross_validation_list)):
        single_split = cross_validation_list[j]
        for k in range(len(single_split)):
            movie_name = single_split[k]
            # for feature 
            feature_path = ".../path/to/visual_features/dir/" + movie_name + '/*.npy' 
            dirs = glob.glob(feature_path)
            # for label 
            label_path = ".../path/to/get/corresponding/clip/emotional/label/" + viewer + '/' + movie_name + '_subtitle_info.txt'
            data = []
            # load label file
            with open(label_path) as csv_file:
                csv_reader = csv.reader(csv_file,delimiter=',')
                for row in csv_reader:
                    cmb = []
                    cmb.append(row[0])
                    cmb.append(row[-1])
                    data.append(cmb)
            data = np.asarray(data)
            # load feature data
            for s_file in dirs:
                visual_features = np.load(s_file)
                visual_features = stats.zscore(visual_features)
                scene_no = s_file[-13:-4]
                sc_index = np.where(data[:,0]==scene_no)
                valence = str(data[sc_index,1])
                #valence = str(data[sc_index,2]) # for avg_viewer labels
                valence = valence.replace('[','').replace(']','').replace("'",'')
                valence = np.round(float(valence),3)
                if j == 0:
                    train_data.append(visual_features)
                    if valence > 0.0:
                        train_label.append(1)
                    else:
                        train_label.append(0)
                elif j == 1:
                    val_data.append(visual_features)
                    if valence > 0.0:
                        val_label.append(1)
                    else:
                        val_label.append(0)
                elif j == 2:
                    test_data.append(visual_features)
                    if valence > 0.0:
                        test_label.append(1)
                    else:
                        test_label.append(0)
        
    return train_data, train_label, val_data, val_label, test_data, test_label

# function to create train, val and test datasets for all cross_validation sets for curricum learning
def create_train_val_test_datasets_curricum_lr(sets,select_set,viewer):
    set_no = list(sets.values())[select_set]
    train_mv_list = set_no["train_mv"]
    val_mv_list = set_no["val_mv"]
    test_mv_list = set_no["test_mv"]
    cross_validation_list = []
    cross_validation_list.append(train_mv_list)
    cross_validation_list.append(val_mv_list)
    cross_validation_list.append(test_mv_list)
    train_data_1 = []
    train_label_1 = []
    train_data_2 = []
    train_label_2 = []
    train_data_3 = []
    train_label_3 = []    
    val_data = []
    val_label = []
    test_data = []
    test_label = []
    # take train , val and test data for single cross_validation set 
    # get visual features
    for j in range(len(cross_validation_list)):
        single_split = cross_validation_list[j]
        for k in range(len(single_split)):
            movie_name = single_split[k]
            # for feature 
            feature_path = ".../path/to/visual_features/dir/" + movie_name + '/*.npy'
            dirs = glob.glob(feature_path)
            # for label 
            label_path = ".../path/to/get/corresponding/clip/emotional/label/" + viewer + '/' + movie_name + '_subtitle_info.txt'
            data = []
            # load label file
            with open(label_path) as csv_file:
                csv_reader = csv.reader(csv_file,delimiter=',')
                for row in csv_reader:
                    cmb = []
                    cmb.append(row[0])
                    cmb.append(row[-1])
                    data.append(cmb)
            data = np.asarray(data)
            # load feature data
            for s_file in dirs:
                visual_features = np.load(s_file)
                visual_features = stats.zscore(visual_features)
                scene_no = s_file[-13:-4]
                sc_index = np.where(data[:,0]==scene_no)
                valence = str(data[sc_index,1])
                #valence = str(data[sc_index,2]) # for avg_viewer labels
                valence = valence.replace('[','').replace(']','').replace("'",'')
                valence = np.round(float(valence),3)
                # for train data
                if j == 0:
                    # training group 1
                    if (valence >= 0.4 and valence <= 1.0) or (valence <= -0.4 and valence >= -1.0):
                        train_data_1.append(visual_features)
                        if valence > 0.0:
                            train_label_1.append(1)
                        else:
                            train_label_1.append(0)
                    # training group 2
                    elif (valence >= 0.1 and valence <= 0.4) or (valence <= -0.1 and valence >= -0.4):
                        train_data_2.append(visual_features)
                        if valence > 0.0:
                            train_label_2.append(1)
                        else:
                            train_label_2.append(0)
                    # training group 3
                    elif (valence >= 0.0 and valence <= 0.1) or (valence <= -0.0 and valence >= -0.1):
                        train_data_3.append(visual_features)
                        if valence > 0.0:
                            train_label_3.append(1)
                        else:
                            train_label_3.append(0)
                # for val data
                elif j == 1:
                    val_data.append(visual_features)
                    if valence > 0.0:
                        val_label.append(1)
                    else:
                        val_label.append(0)
                # for test data
                elif j == 2:
                    test_data.append(visual_features)
                    if valence > 0.0:
                        test_label.append(1)
                    else:
                        test_label.append(0)

    return train_data_1, train_label_1, train_data_2, train_label_2, train_data_3, train_label_3, val_data, val_label, test_data, test_label

# function to create train, val and test datasets for all cross_validation sets 
def create_train_val_test_datasets_mt(sets,select_set,viewers_list,select_viewer):
    set_no = list(sets.values())[select_set]
    train_mv_list = set_no["train_mv"]
    val_mv_list = set_no["val_mv"]
    test_mv_list = set_no["test_mv"]
    cross_validation_list = []
    cross_validation_list.append(train_mv_list)
    cross_validation_list.append(val_mv_list)
    cross_validation_list.append(test_mv_list)
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []
    # take train , val and test data for single cross_validation set 
    # get visual features
    for j in range(len(cross_validation_list)):
        single_split = cross_validation_list[j]
        for k in range(len(single_split)):
            movie_name = single_split[k]
            # for feature 
            feature_path = ".../path/to/visual_features/dir/" + movie_name + '/*.npy'
            dirs = glob.glob(feature_path)
            # load feature data
            for s_file in dirs:
                visual_features = np.load(s_file)
                scene_no = s_file[-13:-4]
                # Train data
                if j == 0:
                    mv_label = []
                    for viewer in range(len(viewers_list)):
                        s_viewer = viewers_list[viewer]
                        # for label 
                        label_path = ".../path/to/get/corresponding/clip/emotional/label/" + s_viewer + '/' + movie_name + '_subtitle_info.txt'
                        data = []
                        # load label file
                        with open(label_path) as csv_file:
                            csv_reader = csv.reader(csv_file,delimiter=',')
                            for row in csv_reader:
                                cmb = []
                                cmb.append(row[0])
                                cmb.append(row[-1])
                                data.append(cmb)
                        data = np.asarray(data)
                        sc_index = np.where(data[:,0]==scene_no)
                        valence = str(data[sc_index,1])
                        valence = valence.replace('[','').replace(']','').replace("'",'')
                        valence = np.round(float(valence),3)
                        if valence > 0.0:
                            mv_label.append(1)
                        else:
                            mv_label.append(0)
                    train_data.append(visual_features)
                    train_label.append(mv_label)
                # Val data
                elif j == 1:
                    s_viewer = viewers_list[select_viewer]
                    # for label 
                    label_path = ".../path/to/get/corresponding/clip/emotional/label/" + s_viewer + '/' + movie_name + '_subtitle_info.txt'
                    data = []
                    # load label file
                    with open(label_path) as csv_file:
                        csv_reader = csv.reader(csv_file,delimiter=',')
                        for row in csv_reader:
                            cmb = []
                            cmb.append(row[0])
                            cmb.append(row[-1])
                            data.append(cmb)

                    data = np.asarray(data)
                    sc_index = np.where(data[:,0]==scene_no)
                    valence = str(data[sc_index,1])
                    valence = valence.replace('[','').replace(']','').replace("'",'')
                    valence = np.round(float(valence),3)
                    val_data.append(visual_features)
                    if valence > 0.0:
                        val_label.append(1)
                    else:
                        val_label.append(0)
                # Test data
                elif j == 2:
                    s_viewer = viewers_list[select_viewer]
                    # for label 
                    label_path = ".../path/to/get/corresponding/clip/emotional/label/" + s_viewer + '/' + movie_name + '_subtitle_info.txt'
                    data = []
                    # load label file
                    with open(label_path) as csv_file:
                        csv_reader = csv.reader(csv_file,delimiter=',')
                        for row in csv_reader:
                            cmb = []
                            cmb.append(row[0])
                            cmb.append(row[-1])
                            data.append(cmb)
                    data = np.asarray(data)
                    sc_index = np.where(data[:,0]==scene_no)
                    valence = str(data[sc_index,1])
                    valence = valence.replace('[','').replace(']','').replace("'",'')
                    valence = np.round(float(valence),3)
                    test_data.append(visual_features)
                    if valence > 0.0:
                        test_label.append(1)
                    else:
                        test_label.append(0)
         
    return train_data, train_label, val_data, val_label, test_data, test_label

# function to create train and test datasets (baseline_split) for all cross_validation sets 
def create_train_val_test_datasets_mt1(sets,viewers_list,select_viewer):
    train_mv_list = sets["train_mv"]
    test_mv_list = sets["test_mv"]
    mv_split_list = []
    mv_split_list.append(train_mv_list)
    mv_split_list.append(test_mv_list)
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    # take train and test data #
    # get visual features
    for j in range(len(mv_split_list)):
        single_split = mv_split_list[j]
        for k in range(len(single_split)):
            movie_name = single_split[k]
            # for feature 
            feature_path = ".../path/to/visual_features/dir/" + movie_name + '/*.npy'                                      
            dirs = glob.glob(feature_path)
            # load feature data
            for s_file in dirs:
                visual_features = np.load(s_file)
                scene_no = s_file[-13:-4]
                # Train data
                if j == 0:
                    mv_label = []
                    for viewer in range(len(viewers_list)):
                        s_viewer = viewers_list[viewer]
                        # for label 
                        label_path = ".../path/to/get/corresponding/clip/emotional/label/" + s_viewer + '/' + movie_name + '_subtitle_info.txt'  
                        data = []
                        # load label file
                        with open(label_path) as csv_file:
                            csv_reader = csv.reader(csv_file,delimiter=',')
                            for row in csv_reader:
                                cmb = []
                                cmb.append(row[0])
                                cmb.append(row[-1])
                                data.append(cmb)
                        data = np.asarray(data)
                        sc_index = np.where(data[:,0]==scene_no)
                        valence = str(data[sc_index,1])
                        valence = valence.replace('[','').replace(']','').replace("'",'')
                        valence = np.round(float(valence),3)
                        if valence > 0.0:
                            mv_label.append(1)
                        else:
                            mv_label.append(0)
                    train_data.append(visual_features)
                    train_label.append(mv_label) 
                # Test data
                elif j == 1:
                    s_viewer = viewers_list[select_viewer]
                    # for label 
                    label_path = ".../path/to/get/corresponding/clip/emotional/label/" + s_viewer + '/' + movie_name + '_subtitle_info.txt'
                    data = []
                    # load label file
                    with open(label_path) as csv_file:
                        csv_reader = csv.reader(csv_file,delimiter=',')
                        for row in csv_reader:
                            cmb = []
                            cmb.append(row[0])
                            cmb.append(row[-1])
                            data.append(cmb)
                    data = np.asarray(data)
                    sc_index = np.where(data[:,0]==scene_no)
                    valence = str(data[sc_index,1])
                    valence = valence.replace('[','').replace(']','').replace("'",'')                                                                                   
                    valence = np.round(float(valence),3)
                    test_data.append(visual_features)
                    if valence > 0.0:
                        test_label.append(1)
                    else:
                        test_label.append(0)
    return train_data, train_label, test_data, test_label

# function for taking valence emotion values 
def get_valence(sentiment):
  valence_emotions = []
  for i in range(len(sentiment)):
    row = sentiment[i]
    valence = row[0]
    valence_emotions.append(valence)
  return valence_emotions

# function for real emotion values into binary 
def get_binary(emotions):
  binary_emo = []
  for i in range(len(emotions)):
    emo = emotions[i]
    if emo > 0:
      binary_emo.append(1)
    else:
      binary_emo.append(0)
  return binary_emo

# function for shuffling 
def get_shuffle_batch(samples,labels):
  labeled_examples = list(zip(samples, labels))
  shuffle(labeled_examples)
  # Separate and return the features and labels.
  features = [example for (example, _) in labeled_examples]
  labels = [label for (_, label) in labeled_examples]
  return features, labels

# function to convert real values of model prediction into binary 
def real_to_binary(predictions):
  md_pred = []
  for i in range(len(predictions)):
    value = predictions[i]
    if value > 0:
      md_pred.append(1)
    else:
      md_pred.append(0)
  return md_pred

# function to get 1D vector
def get_vector(array):
  full_vector = []
  for i in range(len(array)):
    row = array[i]
    for j in range(len(row)):
      value = row[j]
      full_vector.append(value)
  return full_vector

# Function for model performance  
def md_performance(y_actual, y_pred):
  TP = 0
  FP = 0
  TN = 0
  FN = 0
  for i in range(len(y_pred)): 
      if y_actual[i]==y_pred[i]==1:
          TP += 1
      if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
          FP += 1
      if y_actual[i]==y_pred[i]==0:
          TN += 1
      if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
          FN += 1
  accuracy = (TP + TN) / (TP+FN+TN+FP)
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  f1 = 2 * ((precision * recall) / (precision + recall))
  return {
      "accuracy": accuracy,
      "f1_score": f1,
      "precision": precision,
      "recall": recall,
      "true_positives": TP,
      "true_negatives": TN,
      "false_positives": FP,
      "false_negatives": FN
      }
  
# Multilayer perceptron
# Create conv model
def cmb_fc(input_tensor):
    with tf.variable_scope('cmb_fc'):
        flatten_layer = tf.contrib.layers.flatten(input_tensor)
        # fully connected 
        fc1 = tf.contrib.layers.fully_connected(flatten_layer, 2048, activation_fn=tf.nn.relu,
        weights_initializer=tf.truncated_normal_initializer(seed=10),
        biases_initializer=tf.zeros_initializer(),
        trainable=True)
        # fully connected
        fc2 = tf.contrib.layers.fully_connected(fc1, 1024, activation_fn=tf.nn.relu,
        weights_initializer=tf.truncated_normal_initializer(seed=10),
        biases_initializer=tf.zeros_initializer(),
        trainable=True)
    return fc2

# personalized model 1
def personalized_model_1(tensor,keep_prob):
    with tf.variable_scope('p_1'):
      # fully fonnected #
      fc1 = tf.contrib.layers.fully_connected(tensor, 1024, activation_fn=tf.nn.relu,                                                                             
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1, 512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(fc2, 256, activation_fn=tf.nn.relu,                                                                                
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)

      return out

# personalized model 2
def personalized_model_2(tensor,keep_prob):
    with tf.variable_scope('p_2'):
      # fully fonnected #
      fc1 = tf.contrib.layers.fully_connected(tensor, 1024, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1, 512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(fc2, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)

      return out

# personalized model 3
def personalized_model_3(tensor,keep_prob):
    with tf.variable_scope('p_3'):
      # fully fonnected #
      fc1 = tf.contrib.layers.fully_connected(tensor, 1024, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1, 512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(fc2, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)

      return out

# personalized model 4
def personalized_model_4(tensor,keep_prob):
    with tf.variable_scope('p_4'):
      # fully fonnected #
      fc1 = tf.contrib.layers.fully_connected(tensor, 1024, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1, 512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(fc2, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)

      return out

# personalized model 5
def personalized_model_5(tensor,keep_prob):
    with tf.variable_scope('p_5'):
      # fully fonnected #
      fc1 = tf.contrib.layers.fully_connected(tensor, 1024, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1, 512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(fc2, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)

      return out

# personalized model 6
def personalized_model_6(tensor,keep_prob):
    with tf.variable_scope('p_6'):
      # fully fonnected #
      fc1 = tf.contrib.layers.fully_connected(tensor, 1024, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1, 512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(fc2, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)

      return out

# personalized model 7
def personalized_model_7(tensor,keep_prob):
    with tf.variable_scope('p_7'):
      # fully fonnected #
      fc1 = tf.contrib.layers.fully_connected(tensor, 1024, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1, 512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(fc2, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)

      return out

# personalized model 8
def personalized_model_8(tensor,keep_prob):
    with tf.variable_scope('p_8'):
      # fully fonnected #
      fc1 = tf.contrib.layers.fully_connected(tensor, 1024, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1, 512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(fc2, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)

      return out


