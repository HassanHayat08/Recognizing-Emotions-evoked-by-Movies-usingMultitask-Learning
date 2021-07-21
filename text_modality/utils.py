# Recognizing Emotions evoked by Movies using Multitask Learning #
# Hassan Hayat, Carles Ventura, Agata Lapedriza #
# This script contains the supporting functions #  

# import libraries
import numpy as np
from random import shuffle
import re, string
import csv
from math import log
import functools
import random
import collections
import os
import tensorflow as tf

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
def create_train_val_test_datasets(sets,annotator_no):
    all_sets_train_data = []
    all_sets_val_data = []
    all_sets_test_data = []
    # all cross validation sets 
    for i in range(len(sets)):
        set_no = list(sets.values())[i]
        train_mv_list = set_no["train_mv"]
        val_mv_list = set_no["val_mv"]
        test_mv_list = set_no["test_mv"]
        cross_validation_list = []
        cross_validation_list.append(train_mv_list) 
        cross_validation_list.append(val_mv_list)
        cross_validation_list.append(test_mv_list)
        train_data = []
        val_data = []
        test_data = []
        # take train , val and test data for single cross_validation set 
        for j in range(len(cross_validation_list)):
            single_split = cross_validation_list[j]
            for k in range(len(single_split)):
                movie_name = single_split[k]
                movie_path = "/home/hhassan/shared/multitask_learning/data/" + annotator_no + "/" + movie_name + ".txt"
                with open(movie_path) as csv_file:
                    csv_reader = csv.reader(csv_file)
                    if j == 0:
                        for row in csv_reader:
                            train_data.append(row)
                    elif j == 1:
                        for row in csv_reader:
                            val_data.append(row)
                    elif j == 2:
                        for row in csv_reader:
                            test_data.append(row)
        all_sets_train_data.append(train_data)
        all_sets_val_data.append(val_data)
        all_sets_test_data.append(test_data)
    return all_sets_train_data, all_sets_val_data, all_sets_test_data

# function to get vocabulary
def get_vocabulary(names):
  data = []
  clean_text = []
  all_words = []
  annotator_no = "viewer_1"
  for i in range(len(names)):
    mv_name = names[i]
    movie_path = ".../path/to/the/whole/text/data/" + annotator_no + "/" + mv_name + ".txt"
    with open(movie_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
          data.append(row)
  for j in range(len(data)):
    row = data[j]
    string = row[0]
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
    string = string = ''.join([i for i in string if not i.isdigit()])
    clean_text.append(string)
  for k in range(len(clean_text)):
    sentence = clean_text[k]
    no_of_words = sentence.split()
    for m in range(len(no_of_words)):
      single_word = no_of_words[m]
      all_words.append(single_word)
  all_words.append("pad")
  words = [word for word in all_words if word.isalpha()]
  vocab = {y:v for v,y in enumerate(np.unique(words))}
  return vocab

# function for text normalization 
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

# function for padd one word to the same length for all the sentences 
def sentence_padding(text,max_no_words):
  padd_sentence = []
  for j in range(len(text)):
    sentence = text[j]
    words = sentence.split(" ")
    for _ in range(max_no_words - len(words)):
      sentence =  sentence + " " + "pad" 
    padd_sentence.append(sentence)
  return padd_sentence

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

# function for shuffling 
def get_shuffle_batch(samples,labels):
  labeled_examples = list(zip(samples, labels))
  shuffle(labeled_examples)
  # separate and return the features and labels.
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
    row = np.squeeze(row)
    for j in range(len(row)):
      value = row[j]
      full_vector.append(value)
  return full_vector

# function for model performance  
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
  
# function to calculate hinge loss
def Hinge(yHat, y):
    return np.max(0, 1 - yHat * y)

# function to calculate cross-entropy
def CrossEntropy(yHat, y):
    if y == 1:
      return -log(yHat)
    else:
      return -log(1 - yHat)

# function to get positive and negative samples 
def get_train_test_data(text,sentiments):
  data = {}
  data['sentence'] = []
  data['polarity'] = []
  for i in range(len(text)):
    data['sentence'].append(text[i])
    if sentiments[i][0] > 0:
      data['polarity'].append(1)
    else:
      data['polarity'].append(0) 
  return pd.DataFrame.from_dict(data)

# function to take random samples
def take_random_examples(data,pos_examples,neg_examples):
  rand_loc = []
  pos_data = {}
  pos_data['sentence'] = []
  pos_data['polarity'] = []
  for x in range(neg_examples):
    rand_loc.append(random.randint(1,pos_examples-1))
  for y in range(len(rand_loc)):
    loc = rand_loc[y]
    sent = data.iloc[loc,0]
    pol = data.iloc[loc,1]
    pos_data['sentence'].append(sent)
    pos_data['polarity'].append(pol)
  pos_data = pd.DataFrame.from_dict(pos_data)
  frames = [pos_data,neg_data]
  train_data = pd.concat(frames)
  # shuffle 
  train_data = train_data.sample(frac=1).reset_index(drop=True)
  return train_data

# function for data cleaning
def data_cleaning(data):
    all_text = []
    all_label = []
    for i in range(len(data)):
        single_data = data[i]
        text = []
        label = []
        for j in range(len(single_data)):
          row = single_data[j]
          string = row[0]
          val = float(row[-2])
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
          text.append(string)
          if val > 0:
              label.append(1)
          else:
              label.append(0)
    
        all_text.append(text)
        all_label.append(label)
    return all_text, all_label

# function to get max-words
def max_length_of_sentence(data1,data2,data3):
  max_len_data1 = []
  max_len_data2 = []
  max_len_data3 = []
  for i in range(len(data1)):
    single_data = data1[i]
    for ii in range(len(single_data)):
      row = single_data[ii]
      row = row.split(" ")
      max_len_data1.append(len(row))
  for j in range(len(data2)):
    single_data = data2[j]
    for jj in range(len(single_data)):
      row = single_data[jj]
      row = row.split(" ")
      max_len_data2.append(len(row))
  for k in range(len(data3)):
    single_data = data3[k]
    for kk in range(len(single_data)):
      row = single_data[kk]
      row = row.split(" ")
      max_len_data3.append(len(row))
  max_1 = np.max(max_len_data1)
  max_2 = np.max(max_len_data2)
  max_3 = np.max(max_len_data3)
  max_length = [max_1,max_2,max_3]
  return np.max(max_length)

# data padding 
def data_padding(array,max_length):
  data_samples = []
  for i in range(len(array)):
    text = []
    data = array[i]
    for j in range(len(data)):
      sentence = data[j]
      words = sentence.split(" ")
      for _ in range(max_length - len(words)):
        sentence =  sentence + " " + "pad" 
      text.append(sentence)
    data_samples.append(text)
  return data_samples  

# function to load single task learning 
def load_data(mv_list,annotators_list):
  whole_data = []
  for i in range(len(annotators_list)):
    single_data = []
    single_annotator = annotators_list[i]
    for j in range(len(mv_list)):
      mv_name = mv_list[j]
      movie_path = ".../path/to/get/dataset/" + single_annotator + "/" + mv_name + ".txt"
      with open(movie_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
          single_data.append(row)
    whole_data.append(single_data)
  return whole_data

# function to get avg predictions
def avg_annotations_pred(data):
  x1 , y1 = np.shape(data)
  all_test_pred = np.zeros(shape=(y1, x1))
  new_data = np.transpose(data)
  avg_pred = []
  for i in range(len(new_data)):
    row = new_data[i]
    neg_count = len(list(filter(lambda x: (x < 1), row))) 
    pos_count = len(list(filter(lambda x: (x > 0), row)))
    if pos_count > neg_count :
      avg_pred.append(1)
    else:
      avg_pred.append(0)
  return avg_pred

# function to get correct and incorrect prediction of test models
def get_correct_inncorrect_sentences(data,gt_value,model_pred):
    correct_sentence = {}
    correct_sentence['sentence'] = []
    correct_sentence['ground_truth'] = []
    correct_sentence['model_pred'] = []
    incorrect_sentence = {}
    incorrect_sentence['sentence'] = []
    incorrect_sentence['ground_truth'] = []
    incorrect_sentence['model_pred'] = []
    for i in range(len(data)):
        sentence = data[i]
        sentence_gt = gt_value[i]
        sentence_pred = model_pred[i]
        if sentence_gt == sentence_pred:
            correct_sentence['sentence'].append(sentence)
            correct_sentence['ground_truth'].append(sentence_gt)
            correct_sentence['model_pred'].append(sentence_pred)
        else:
            incorrect_sentence['sentence'].append(sentence)
            incorrect_sentence['ground_truth'].append(sentence_gt)
            incorrect_sentence['model_pred'].append(sentence_pred)
    return correct_sentence , incorrect_sentence

# function to get unpad sentence 
def get_unpad_sentence(data):
    sentence = []
    for i in range(len(data)):
        single_batch = data[i]
        for j in range(len(single_batch)):
            text = single_batch[j]
            text = text.replace('pad','')
            text = " ".join(text.split())
            sentence.append(text)
    return sentence

# function for 2D convolution
def conv2d(X, k_size, b):
    # conv2D wrapper, with bias and relu activation 
    # VALID means No Padding and SAME means with padding 
    X = tf.layers.conv2d(inputs=X, filters=300, kernel_size=k_size, strides=[3,3], padding="SAME",
                         activation=None,
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                         trainable=True)
    X = tf.nn.bias_add(X, b)
    return tf.nn.relu(X)

# function for 2D max-pooling
def maxpool2d(X,k):
    # max-pool2D wrapper
    return tf.nn.max_pool(X, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

# create conv model
def conv_net(x, VOCAB_LEN, EMBED_SIZE, biases):
    with tf.variable_scope('conv'):
      # embedding layer ###
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

# personalized model 1
def personalized_model_1(tensor,keep_prob):
    with tf.variable_scope('p_1'):
      # fully connected #
      fc1 = tf.contrib.layers.fully_connected(tensor ,1024, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1 ,512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_1 = tf.nn.dropout(fc2, keep_prob)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(drop_out_1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_2 = tf.nn.dropout(fc3, keep_prob)
      # fully connected
      out = tf.contrib.layers.fully_connected(drop_out_2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      return out

# personalized model 2
def personalized_model_2(tensor,keep_prob):
    with tf.variable_scope('p_2'):
      # fully connected #
      fc1 = tf.contrib.layers.fully_connected(tensor ,1024, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1 ,512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_1 = tf.nn.dropout(fc2, keep_prob)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(drop_out_1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_2 = tf.nn.dropout(fc3, keep_prob)
      # fully connected
      out = tf.contrib.layers.fully_connected(drop_out_2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      return out

# personalized model 3
def personalized_model_3(tensor,keep_prob):
    with tf.variable_scope('p_3'):
      # fully connected #
      fc1 = tf.contrib.layers.fully_connected(tensor ,1024, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1 ,512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_1 = tf.nn.dropout(fc2, keep_prob)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(drop_out_1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_2 = tf.nn.dropout(fc3, keep_prob)
      # fully connected
      out = tf.contrib.layers.fully_connected(drop_out_2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      return out

# personalized model 4
def personalized_model_4(tensor,keep_prob):
    with tf.variable_scope('p_4'):
      # fully connected #
      fc1 = tf.contrib.layers.fully_connected(tensor ,1024, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1 ,512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_1 = tf.nn.dropout(fc2, keep_prob)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(drop_out_1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_2 = tf.nn.dropout(fc3, keep_prob)
      # fully connected
      out = tf.contrib.layers.fully_connected(drop_out_2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      return out

# personalized model 5
def personalized_model_5(tensor,keep_prob):
    with tf.variable_scope('p_5'):
      # fully connected #
      fc1 = tf.contrib.layers.fully_connected(tensor ,1024, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1 ,512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_1 = tf.nn.dropout(fc2, keep_prob)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(drop_out_1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_2 = tf.nn.dropout(fc3, keep_prob)
      # fully connected
      out = tf.contrib.layers.fully_connected(drop_out_2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      return out

# personalized model 6
def personalized_model_6(tensor,keep_prob):
    with tf.variable_scope('p_6'):
      # fully connected #
      fc1 = tf.contrib.layers.fully_connected(tensor ,1024, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1 ,512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_1 = tf.nn.dropout(fc2, keep_prob)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(drop_out_1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_2 = tf.nn.dropout(fc3, keep_prob)
      # fully connected
      out = tf.contrib.layers.fully_connected(drop_out_2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      return out

# personalized model 7
def personalized_model_7(tensor,keep_prob):
    with tf.variable_scope('p_7'):
      # fully connected #
      fc1 = tf.contrib.layers.fully_connected(tensor ,1024, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1 ,512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_1 = tf.nn.dropout(fc2, keep_prob)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(drop_out_1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_2 = tf.nn.dropout(fc3, keep_prob)
      # fully connected
      out = tf.contrib.layers.fully_connected(drop_out_2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      return out

# personalized model 8
def personalized_model_8(tensor,keep_prob):
    with tf.variable_scope('p_8'):
      # fully connected #
      fc1 = tf.contrib.layers.fully_connected(tensor ,1024, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # fully connected
      fc2 = tf.contrib.layers.fully_connected(fc1 ,512, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_1 = tf.nn.dropout(fc2, keep_prob)
      # fully connected
      fc3 = tf.contrib.layers.fully_connected(drop_out_1, 256, activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      # dropout
      drop_out_2 = tf.nn.dropout(fc3, keep_prob)
      # fully connected
      out = tf.contrib.layers.fully_connected(drop_out_2, 1, activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(seed=10),
      biases_initializer=tf.zeros_initializer(),
      trainable=True)
      return out

