# Recognizing Emotions evoked by Movies using Multitask Learning #
# Hassan Hayat, Carles Ventura, Agata Lapedriza #
# This script contains the supported functions for Single-Task and Multi-Task using Single-Modality and Multi-Modality #

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
import re

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

# function to create train, val and test datasets for all cross_validation sets #
def create_train_val_test_datasets_st_joint(sets,select_set,viewer):
    set_no = list(sets.values())[select_set]
    train_mv_list = set_no["train_mv"]
    val_mv_list = set_no["val_mv"]
    test_mv_list = set_no["test_mv"]
    cross_validation_list = []
    cross_validation_list.append(train_mv_list)
    cross_validation_list.append(val_mv_list)
    cross_validation_list.append(test_mv_list)
    # visual
    visual_train_data = []
    train_label = []
    visual_val_data = []
    val_label = []
    visual_test_data = []
    test_label = []
    # text
    text_train_data = []
    text_val_data = []
    text_test_data = []
    # audio
    audio_train_data = [] 
    audio_val_data = []
    audio_test_data = []
    # take train , val and test data for single cross_validation set 
    # get visual features 
    for j in range(len(cross_validation_list)):
        single_split = cross_validation_list[j]
        for k in range(len(single_split)):
            movie_name = single_split[k]
            # for feature 
            path_visual = ".../path/to/visual_features/dir/" + movie_name + '/*.npy' 
            dirs = glob.glob(path_visual)
            # for label 
            label_path = ".../path/to/get/corresponding/clip/emotional/label/" + viewer + '/' + movie_name + '_subtitle_info.txt'
            data = []
            # load label file
            with open(label_path) as csv_file:
                csv_reader = csv.reader(csv_file,delimiter=',')
                for row in csv_reader:
                    cmb = []
                    cmb.append(row[0])
                    cmb.append(row[1])
                    cmb.append(row[-1])
                    data.append(cmb)

            data = np.asarray(data)
            # load feature data
            for s_file in dirs:
                visual_features = np.load(s_file)
                visual_features = stats.zscore(visual_features)
                scene_no = s_file[-13:-4]
                audio_file = '.../path/to/corresponding/audio_features/dir/' + movie_name + '/' + scene_no + '.npy'
                audio_features = np.load(audio_file)
                sc_index = np.where(data[:,0]==scene_no)
                valence = str(data[sc_index,-1]) # for single viewer
                #valence = str(data[sc_index,2]) # for avg_viewer labels
                valence = valence.replace('[','').replace(']','').replace("'",'')
                valence = np.round(float(valence),3)
                subtitle = str(data[sc_index,1])
                if j == 0:
                    visual_train_data.append(visual_features)
                    audio_train_data.append(audio_features.flatten())
                    text_train_data.append(subtitle)
                    if valence > 0.0:
                        train_label.append(1)
                    else:
                        train_label.append(0)
                elif j == 1:
                    visual_val_data.append(visual_features)
                    audio_val_data.append(audio_features.flatten())
                    text_val_data.append(subtitle)
                    if valence > 0.0:
                        val_label.append(1)
                    else:
                        val_label.append(0)
                elif j == 2:
                    visual_test_data.append(visual_features)
                    audio_test_data.append(audio_features.flatten())
                    text_test_data.append(subtitle)
                    if valence > 0.0:
                        test_label.append(1)
                    else:
                        test_label.append(0)

        
    return visual_train_data, text_train_data, audio_train_data, train_label, visual_val_data, text_val_data, audio_val_data, val_label, visual_test_data, text_test_data, audio_test_data, test_label

# function to check if string contains any integer data
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

# function to convert list type to string data
def listToString(s): 
    # initialize an empty string
    str1 = " " 
    # return string  
    return (str1.join(s))
        
# function to create train, val and test datasets for all cross_validation sets #
def create_train_val_test_datasets_mt_joint(sets,select_set,viewers_list,viewer_no):
    set_no = list(sets.values())[select_set]
    train_mv_list = set_no["train_mv"]
    val_mv_list = set_no["val_mv"]
    test_mv_list = set_no["test_mv"]
    cross_validation_list = []
    cross_validation_list.append(train_mv_list)
    cross_validation_list.append(val_mv_list)
    cross_validation_list.append(test_mv_list)
    # visual
    visual_train_data = []
    train_label = []
    visual_val_data = []
    val_label = []
    visual_test_data = []
    test_label = []
    # text
    text_train_data = []
    text_val_data = []
    text_test_data = []
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
                                string = []
                                number = []
                                new_row = []
                                for item in row:
                                    if hasNumbers(item):
                                        number.append(item)
                                    else:
                                        string.append(item)
                                string = listToString(string)
                                new_row.append(number[0])
                                new_row.append(string)
                                new_row.append(number[-1])
                                data.append(new_row)

                        data = np.asarray(data)
                        sc_index = np.where(data[:,0]==scene_no)
                        valence = str(data[sc_index,2])
                        valence = valence.replace('[','').replace(']','').replace("'",'')                                                                               
                        valence = np.round(float(valence),3)
                        subtitle = str(data[sc_index,1])
                        if valence > 0.0:
                            mv_label.append(1)
                        else:
                            mv_label.append(0)
                    text_train_data.append(subtitle)
                    visual_train_data.append(visual_features)
                    train_label.append(mv_label)
                # Val data
                if j == 1:
                    s_viewer = viewers_list[viewer_no]
                    # for label 
                    label_path = ".../path/to/get/corresponding/clip/emotional/label/" + s_viewer + '/' + movie_name + '_subtitle_info.txt'
                    data = []
                    # load label file
                    with open(label_path) as csv_file:
                        csv_reader = csv.reader(csv_file,delimiter=',')
                        for row in csv_reader:
                            string = []
                            number = []
                            new_row = []
                            for item in row:
                                if hasNumbers(item):
                                    number.append(item)
                                else:
                                    string.append(item)
                            string = listToString(string)
                            new_row.append(number[0])
                            new_row.append(string)
                            new_row.append(number[-1])
                            data.append(new_row)
                    data = np.asarray(data)
                    sc_index = np.where(data[:,0]==scene_no)
                    valence = str(data[sc_index,2])
                    valence = valence.replace('[','').replace(']','').replace("'",'')
                    valence = np.round(float(valence),3)
                    subtitle = str(data[sc_index,1])
                    if valence > 0.0:
                        val_label.append(1)
                    else:
                        val_label.append(0)
                    text_val_data.append(subtitle)
                    visual_val_data.append(visual_features)
                # Test data
                if j == 2:
                    s_viewer = viewers_list[viewer_no]
                    # for label 
                    label_path = ".../path/to/get/corresponding/clip/emotional/label/" + s_viewer + '/' + movie_name + '_subtitle_info.txt'
                    data = []
                    # load label file
                    with open(label_path) as csv_file:
                        csv_reader = csv.reader(csv_file,delimiter=',')
                        for row in csv_reader:
                            string = []
                            number = []
                            new_row = []
                            for item in row:
                                if hasNumbers(item):
                                    number.append(item)
                                else:
                                    string.append(item)
                            string = listToString(string)
                            new_row.append(number[0])
                            new_row.append(string)
                            new_row.append(number[-1])
                            data.append(new_row)
                    data = np.asarray(data)
                    sc_index = np.where(data[:,0]==scene_no)
                    valence = str(data[sc_index,2])
                    valence = valence.replace('[','').replace(']','').replace("'",'')
                    valence = np.round(float(valence),3)
                    subtitle = str(data[sc_index,1])
                    if valence > 0.0:
                        test_label.append(1)
                    else:
                        test_label.append(0)
                    text_test_data.append(subtitle)
                    visual_test_data.append(visual_features)
                    
    return visual_train_data, text_train_data, train_label, visual_val_data, text_val_data, val_label, visual_test_data, text_test_data, test_label
    
   
# function to create train, val and test datasets for all cross_validation sets #
def create_train_val_test_datasets_mt_joint1(sets,viewers_list,viewer_no):
    train_mv_list = sets["train_mv"]
    test_mv_list = sets["test_mv"]
    cross_validation_list = []
    cross_validation_list.append(train_mv_list)
    cross_validation_list.append(test_mv_list)
    # visual
    visual_train_data = []
    train_label = []
    visual_test_data = []
    test_label = []
    # text
    text_train_data = []
    text_test_data = []
    audio_train_data =[]
    audio_test_data = []
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
                audio_file = '.../path/to/corresponding/audio_features/dir/' + movie_name + '/' + scene_no + '.npy'
                audio_features = np.load(audio_file)
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
                                string = []
                                number = []
                                new_row = []
                                for item in row:
                                    if hasNumbers(item):
                                        number.append(item)
                                    else:
                                        string.append(item)
                                string = listToString(string)
                                new_row.append(number[0])
                                new_row.append(string)
                                new_row.append(number[-1])
                                data.append(new_row)
                        data = np.asarray(data)
                        sc_index = np.where(data[:,0]==scene_no)
                        valence = str(data[sc_index,2])
                        valence = valence.replace('[','').replace(']','').replace("'",'')                                                      
                        valence = np.round(float(valence),3)
                        subtitle = str(data[sc_index,1])
                        if valence > 0.0:
                            mv_label.append(1)
                        else:
                            mv_label.append(0)
                    text_train_data.append(subtitle)
                    visual_train_data.append(visual_features)
                    audio_train_data.append(audio_features.flatten())
                    train_label.append(mv_label)
                # test data
                if j == 1:
                    s_viewer = viewers_list[viewer_no]
                    # for label 
                    label_path = ".../path/to/get/corresponding/clip/emotional/label/" + s_viewer + '/' + movie_name + '_subtitle_info.txt'
                    data = []
                    # load label file
                    with open(label_path) as csv_file:
                        csv_reader = csv.reader(csv_file,delimiter=',')
                        for row in csv_reader:
                            string = []
                            number = []
                            new_row = []
                            for item in row:
                                if hasNumbers(item):
                                    number.append(item)
                                else:
                                    string.append(item)
                            string = listToString(string)
                            new_row.append(number[0])
                            new_row.append(string)
                            new_row.append(number[-1])
                            data.append(new_row)
                    data = np.asarray(data)
                    sc_index = np.where(data[:,0]==scene_no)
                    valence = str(data[sc_index,2])
                    valence = valence.replace('[','').replace(']','').replace("'",'')
                    valence = np.round(float(valence),3)
                    subtitle = str(data[sc_index,1])
                    if valence > 0.0:
                        test_label.append(1)
                    else:
                        test_label.append(0)
                    text_test_data.append(subtitle)
                    visual_test_data.append(visual_features)
                    audio_test_data.append(audio_features.flatten())
    return visual_train_data, text_train_data, audio_train_data, train_label, visual_test_data, text_test_data, audio_test_data, test_label

# function to clean senetnces
def data_cleaning(data):
    all_text = []
    for i in range(len(data)):
        string = data[i]
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = " ".join([word for word in string.split() if len(word) > 1 and word.isalnum()])
        string = string.lower()
        all_text.append(string)
    return all_text

# data padding 
def data_padding(data,max_length):
  data_samples = []
  for i in range(len(data)):
    sentence = data[i]
    words = sentence.split(" ")
    for _ in range(max_length - len(words)):
        sentence =  sentence + " " + "pad"
    data_samples.append(sentence)
  return data_samples

# function to get the text vocabulary 
def get_vocab(text):
  all_words = []
  for i in range(len(text)):
    sentence = text[i]
    no_of_words = sentence.split()
    for j in range(len(no_of_words)):
      single_word = no_of_words[j]
      all_words.append(single_word)
  all_words.append("pad")
  words = [word for word in all_words if word.isalpha()]
  vocab = {k:v for v,k in enumerate(np.unique(words))}
  return vocab

# function for shuffling 
def get_shuffle_batch(samples_1, samples_2, sample_3, labels):
  labeled_examples = list(zip(samples_1, samples_2, sample_3, labels))
  shuffle(labeled_examples)
  # Separate and return the features and labels
  text = [example1 for (example1,_,_,_) in labeled_examples]
  visual = [example2 for (_,example2,_,_) in labeled_examples]
  audio = [example3 for (_,_,example3,_) in labeled_examples]
  labels = [label for (_,_,_,label) in labeled_examples]
  return text,visual,audio, labels

# function to get word_ids 
def get_word_ids(text,vocab):
  word_ids = []
  for i in range(len(text)):
    sentence = np.zeros((18))
    single_text = text[i]
    single_text = " ".join(single_text.split())
    words = single_text.split(" ")
    for j in range(len(words)):
        single_word = words[j]
        ID = vocab.get(single_word)
        sentence[j] = ID
    word_ids.append(sentence)
  return word_ids


# Function for text normalization 
def get_text_and_emotion(file_path):
  data = []
  with open(file_path) as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        data.append(row)

  sentiments = np.zeros(shape=(len(data), 2))
  movie_subscript = []
  for i in range(len(data)):
    row = str(data[i])
    row = row.split(",")
    sentence = row[0]
    sentence = sentence.replace("[","").replace("'","")
    sentence = re.sub('[!@#$"?-]', '', sentence)
    sentence = " ".join(sentence.split()) # removing spaces from start and end                                                                                     
    sentence = sentence.lower()
    movie_subscript.append(sentence)
    checkWords = ("\Al","\bill","\bhes","\Aim","\bisnt","\byoure","\bhis","\bdidnt","\bive","thats","\bits","\bshes","\bid","dont","whos","aint","theyre","theyll","shouldnt","arent","weve")
    repWords = ("i","i will","he is","i am","is not","you are","he is","did not","i have","that is","it is","she is","i would","do not","who is","am not","they are","they will","should not","are not","we have")
    for check, rep in zip(checkWords, repWords):
        line = sentence.replace(check, rep)
        if line != sentence:
          movie_subscript.pop()
          movie_subscript.append(line)
    val = row[-2]
    val = val.replace("'",'').replace('"','')
    val = float(val)
    aro = row[-1]
    aro = aro.replace(']','').replace("'",'')
    aro = float(aro)
    sentiments[i][0] = val
    sentiments[i][1] = aro

  return movie_subscript, sentiments

# function ofr 2D convolution
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

# function for taking valence emotion values #
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
        # fully connected 1
        fc1 = tf.contrib.layers.fully_connected(input_tensor, 2048, activation_fn=tf.nn.relu,
        weights_initializer=tf.truncated_normal_initializer(),
        biases_initializer=tf.zeros_initializer(),
        trainable=True)
        # fully connected 2
        fc2 = tf.contrib.layers.fully_connected(fc1, 1024, activation_fn=tf.nn.relu,
        weights_initializer=tf.truncated_normal_initializer(),
        biases_initializer=tf.zeros_initializer(),
        trainable=True)
    return fc2

# personalized model 1
def personalized_model_1(tensor,keep_prob):
    with tf.variable_scope('p_1'):
      # fully connected 
      fc1 = tf.contrib.layers.fully_connected(tensor, 512, activation_fn=tf.nn.relu,                                                                             
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected 
      fc2 = tf.contrib.layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected 
      fc3 = tf.contrib.layers.fully_connected(fc2, 128, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout 
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)

      return out

# personalized model 2
def personalized_model_2(tensor,keep_prob):
    with tf.variable_scope('p_2'):
      # fully connected 
      fc1 = tf.contrib.layers.fully_connected(tensor, 512, activation_fn=tf.nn.relu,                                                                            
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected 
      fc2 = tf.contrib.layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(fc2, 128, activation_fn=tf.nn.relu,                                                                                   
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)

      return out

# personalized model 3
def personalized_model_3(tensor,keep_prob):
    with tf.variable_scope('p_3'):
      # fully connected 
      fc1 = tf.contrib.layers.fully_connected(tensor, 512, activation_fn=tf.nn.relu,                                                                             
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected 
      fc3 = tf.contrib.layers.fully_connected(fc2, 128, activation_fn=tf.nn.relu,                                                                                   
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.contrib.layers.xavier_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)

      return out

# personalized model 4
def personalized_model_4(tensor,keep_prob):
    with tf.variable_scope('p_4'):
      # fully fonnected 
      fc1 = tf.contrib.layers.fully_connected(tensor, 512, activation_fn=tf.nn.relu,                                                                             
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(fc2, 128, activation_fn=tf.nn.relu,                                                                                  
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)

      return out

# personalized model 5
def personalized_model_5(tensor,keep_prob):
    with tf.variable_scope('p_5'):
      # fully connected 
      fc1 = tf.contrib.layers.fully_connected(tensor, 512, activation_fn=tf.nn.relu,                                                                             
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(fc2, 128, activation_fn=tf.nn.relu,                                                                                   
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)

      return out

# personalized model 6
def personalized_model_6(tensor,keep_prob):
    with tf.variable_scope('p_6'):
      # fully connected 
      fc1 = tf.contrib.layers.fully_connected(tensor, 512, activation_fn=tf.nn.relu,                                                                             
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(fc2, 128, activation_fn=tf.nn.relu,                                                                                   
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)

      return out

# personalized model 7
def personalized_model_7(tensor,keep_prob):
    with tf.variable_scope('p_7'):
      # fully connected 
      fc1 = tf.contrib.layers.fully_connected(tensor, 512, activation_fn=tf.nn.relu,                                                                             
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(fc2, 128, activation_fn=tf.nn.relu,                                                                                   
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01,scope=None),
      trainable=True)

      return out

# personalized model 8
def personalized_model_8(tensor,keep_prob):
    with tf.variable_scope('p_8'):
      # Fully Connected #
      fc1 = tf.contrib.layers.fully_connected(tensor, 512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(fc2, 128, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      dp2 = tf.nn.dropout(fc3, keep_prob)
      # output
      out = tf.contrib.layers.fully_connected(dp2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)

      return out
                                                                                                       
