import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2

from ths.nn.metrics.f1score import f1, precision, recall, fprate
from ths.nn.sequences.cnn import TweetSentimentInception
from ths.nn.sequences.tweets import TweetSentiment2LSTM2Dense3Layer
from ths.utils.files import GloveEmbedding
from ths.utils.sentences import SentenceToIndices, PadSentences, TrimSentences
from sklearn.model_selection import StratifiedKFold


class ProcessTweetsGloveLSTM2Layer:
    def __init__(self, labeled_tweets_filename, embedding_filename):
        self.labeled_tweets_filename = labeled_tweets_filename
        self.embedding_filename = embedding_filename

    def plot(self, history):
        # summarize history for accuracy
        plt.figure(1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # summarize history for loss
        plt.figure(2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def process(self, json_filename, h5_filename, plot=False, epochs = 100):
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
        G = GloveEmbedding(self.embedding_filename, dimensions=50)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_train_indices, max_len = S.map_sentence_list(X_all)
        print("Train data mappend to indices")
        if max_len % 2 != 0:
            max_len = max_len + 1

        P = PadSentences(max_len)
        X_train_pad = P.pad_list(X_train_indices)
        print("Train data padded")
        X_mapped = np.array(X_train_pad)
        Y_mapped = np.array(Y_all)
        print("Data Converted")
        # divide the data into training, validation, and and test
        num_data = len(X_all)
        test_count = math.floor(num_data*0.20)
        validation_count = test_count
        training_count = num_data - (validation_count + test_count)
        #test data set
        X_test_sentences = X_mapped[0:test_count]
        Y_test = Y_mapped[0:test_count]
        # training and cross validation set
        X_train_cross = X_mapped[test_count:]
        Y_train_cross = Y_mapped[test_count:]

        print("length train_cross data: ", len(X_train_cross))
        #X_validation_sentences = X_all[test_count: validation_count]
        #Y_validation = Y_all[test_count: validation_count]

        print("Data Divided")
        ones_count = np.count_nonzero(Y_train_cross)
        zeros_count = len(Y_train_cross) - ones_count
        print("ones_count: ", ones_count)
        print("zeros_count: ", zeros_count)
        print("Train data convert to numpy arrays")
        # fixed random seed number for reproducibility purposes
        seed = 11
        np.random.seed(seed)
        # 5-fold sets (10-fold was too small)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cross_validation_scores = []
        i = 0
        for train, cv in kfold.split(X_train_cross, Y_train_cross):
            print("BEGIN RUN: ", i)
            print("train: ", train)
            print("cv: ", cv)
            #training data
            X_train = X_train_cross[train] #bitch
            Y_train = Y_train_cross[train]
            #cross validation data
            X_cv = X_train_cross[cv]
            Y_cv  = Y_train_cross[cv]
            NN =TweetSentiment2LSTM2Dense3Layer(max_len, G)
            print("len(train): ", len(X_train))
            print("len(cv): ", len(X_cv))
            print("model created")
            NN.build(first_layer_units = max_len, second_layer_units = max_len, relu_dense_layer=16, dense_layer_units = 1,
                     first_layer_dropout=0, second_layer_dropout=0, third_layer_dropout=0)

            print("model built")
            NN.summary()
            #sgd = SGD(lr=0.03, momentum=0.009, decay=0.001, nesterov=True)
            rmsprop = RMSprop(decay=0.003)
            #adam = Adam(lr=0.1, decay=0.05)
            #sgd = SGD(lr=0.05)
            NN.compile(optimizer=rmsprop, loss="binary_crossentropy", metrics=['accuracy', precision, recall, f1, fprate])
            print("model compiled")
            print("Begin training")
            callback = TensorBoard(log_dir="/tmp/logs")
            history = NN.fit(X_train, Y_train, epochs=epochs, batch_size=32, callbacks=[callback], verbose=2)
            print("Model trained")

            # Test on CROSS VALIDATION DATA
            scores= NN.evaluate(X_cv, Y_cv)
            print("metric names: ", NN.get_model().metrics_names)
            print("accuracy: ", scores)

            #X_test_indices, max_len = S.map_sentence_list(X_validation_sentences)
            # print("Test data mapped")
            # X_test_pad = P.pad_list(X_test_indices)
            # print("Test data padded")
            # X_test = np.array(X_test_pad)
            # Y_test = np.array(Y_test)
            # print("Test data converted to numpy arrays")
            # loss, acc = NN.evaluate(X_test, Y_test, callbacks=[callback])
            # print("accuracy: ", acc)

            print("Storing model and weights")
            json_filename = "trained/lstmkfold" + str(i) + ".json"
            h5_filename = "trained/lstmkfold" + str(i) + ".h5"
            NN.save_model(json_filename, h5_filename)
            #if plot:
            #    print("Ploting")
            #    self.plot(history)
            print("END RUN: ", i)
            i = i + 1
        print("Done!")

