import csv
import math

import matplotlib.pyplot as plt
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D

max_words= 10000

class ProcessTweetsLogistic:
    def __init__(self, labeled_tweets_filename, embedding_filename):
        self.labeled_tweets_filename = labeled_tweets_filename
        self.embedding_filename = embedding_filename

    def process(self, json_filename=None, h5_filename=None, plot=False, epochs = 100):
        np.random.seed(11)
        # open the file with tweets
        X_all = []
        Y_all = []
        All  = []
        with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter = ',')
            for r in csv_file:
                if i !=0:
                    All.append(r)
                i = i + 1

        print("len(All): ", len(All))
        #randoming shuffle all tweets
        np.random.shuffle(All)

        ones_count = 0
        for r in All:
            tweet = r[0].strip()
            label = int(r[1])
            X_all.append(tweet)
            Y_all.append(label)

        print("Data Ingested")
        print("X_all[0]: ", X_all[0])
        tokenizer = Tokenizer(num_words=max_words, oov_token='unk')
        print("Fitting data")
        tokenizer.fit_on_texts(X_all)
        X_Seq_All = tokenizer.texts_to_sequences(X_all)

        print("X_Seq_All[0]", X_Seq_All[0])
        print("Final Conversion")
        X_Train = tokenizer.sequences_to_matrix(X_Seq_All, mode='binary')
        print("train_x[0]", X_Train[0])
        Y_Train = Y_all
        print("Create Model")
        model = Sequential()
        model.add(Dense(1, input_dim=10000))
        model.add(Activation('sigmoid'))
        model.summary()
        print("Compilation")
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_Train, Y_Train, epochs=epochs, validation_split=0.20)
        print("Done")


class ProcessTweetsConv1D:
    def __init__(self, labeled_tweets_filename, embedding_filename):
        self.labeled_tweets_filename = labeled_tweets_filename
        self.embedding_filename = embedding_filename

    def process(self, json_filename=None, h5_filename=None, plot=False, epochs = 100):
        np.random.seed(11)
        # open the file with tweets
        X_all = []
        Y_all = []
        All  = []
        with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter = ',')
            for r in csv_file:
                if i !=0:
                    All.append(r)
                i = i + 1

        print("len(All): ", len(All))
        #randoming shuffle all tweets
        np.random.shuffle(All)

        ones_count = 0
        for r in All:
            tweet = r[0].strip()
            label = int(r[1])
            X_all.append(tweet)
            Y_all.append(label)

        print("Data Ingested")
        print("X_all[0]: ", X_all[0])
        tokenizer = Tokenizer(num_words=max_words, oov_token='unk')
        print("Fitting data")
        tokenizer.fit_on_texts(X_all)
        X_Seq_All = tokenizer.texts_to_sequences(X_all)

        print("X_Seq_All[0]", X_Seq_All[0])
        print("Final Conversion")
        X_Train = tokenizer.sequences_to_matrix(X_Seq_All, mode='binary')
        print("train_x[0]", X_Train[0])
        Y_Train = Y_all
        print("Create Model")
        model = Sequential()
        model.add(Conv1D(filters=5, kernel_size=3, padding='same', input_shape=(10000, 1)))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()
        print("Compilation")
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_Train, Y_Train, epochs=epochs, validation_split=0.20)
        print("Done")



