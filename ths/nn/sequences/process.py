from ths.nn.sequences.tweets import TweetSentiment2LSTM, TweetSentiment3LSTM, TweetSentiment2LSTM2Dense, TweetSentiment2LSTM2DenseSM, TweetSentimentGRUSM, TweetSentimentCNN, TweetSentiment2DCNN, TweetSentiment2DCNNv2, TweetSentiment2DCNNv3, TweetSentiment2DCNNv4, TweetSentiment2LSTM2DenseSMv2
from ths.utils.files import GloveEmbedding, Word2VecEmbedding
from ths.utils.sentences import SentenceToIndices, SentenceToEmbedding, PadSentences, TrimSentences
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.regularizers import l2
from ths.nn.metrics.f1score import f1, precision, recall, fprate

import numpy as np
import csv
import math

class ProcessTweetsGlove:
    def __init__(self, labeled_tweets_filename, embedding_filename):
        self.labeled_tweets_filename = labeled_tweets_filename
        self.embedding_filename = embedding_filename

    def process(self, json_filename, h5_filename):
        np.random.seed(11)
        # open the file with tweets
        X_all = []
        Y_all = []
        with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter = ',')
            for r in csv_file:
                if i !=0:
                    tweet = r[0]
                    label = r[1]
                    X_all.append(tweet)
                    Y_all.append(label)
                i = i + 1
        print("Data Ingested")
        # divide the data into training and test
        num_data = len(X_all)
        limit = math.ceil(num_data * 0.60)
        # divide the data into X_train, Y_train, X_test, Y_test
        X_train_sentences = X_all[0: limit]
        Y_train = Y_all[0: limit]
        X_test_sentences = X_all[limit:]
        Y_test = Y_all[limit:]
        print("Data Divided")
        #Get embeeding
        G = GloveEmbedding(self.embedding_filename)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_train_indices, max_len  = S.map_sentence_list(X_train_sentences)
        print("Train data mappend to indices")
        P = PadSentences(max_len)
        X_train_pad = P.pad_list(X_train_indices)
        print("Train data padded")
        # Trim
        #trim_size = 40
        #Trim = TrimSentences(trim_size)
        #X_train_pad = Trim.trim_list(X_train_pad)
        #convert to numPY arrays
        X_train = np.array(X_train_pad)
        Y_train = np.array(Y_train)
        print("Train data convert to numpy arrays")
        NN = TweetSentiment2LSTM2Dense(max_len, G)
        #NN = TweetSentiment2LSTM2Dense(trim_size, G)

        print("model created")
        NN.build(first_layer_units = 128, dense_layer_units=1, first_layer_dropout=0, second_layer_dropout=0)
        print("model built")
        NN.summary()
        sgd = SGD(lr=0.3, momentum=0.001, decay=0.01, nesterov=False)
        adam = Adam(lr=0.03)
        #NN.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer=adam)
        NN.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer='rmsprop')

        print("model compiled")
        print("Begin training")
        callback = TensorBoard(log_dir="/tmp/logs")
        NN.fit(X_train, Y_train, epochs=5, callbacks=[callback])
        print("Model trained")
        X_test_indices, max_len = S.map_sentence_list(X_test_sentences)
        print("Test data mapped")
        X_test_pad = P.pad_list(X_test_indices)
        print("Test data padded")
        X_test = np.array(X_test_pad)
        Y_test = np.array(Y_test)
        print("Test data converted to numpy arrays")
        loss, acc = NN.evaluate(X_test, Y_test)
        print("accuracy: ", acc, ", loss: " , loss)
        T = "I have a bad case of vomit"
        X_Predict = ["my zika is bad", "i love colombia", "my has been tested for ebola", "there is a diarrhea outbreak in the city"]
        X_Predict_Idx, max_len2 = S.map_sentence_list(X_Predict)
        i =0
        for s in X_Predict_Idx:
            print(str(i)+ ": ", s)
            i = i + 1
        print(X_Predict)
        X_Predict_Final = P.pad_list(X_Predict_Idx)
        #X_Predict_Final = Trim.trim_list(X_Predict_Final)
        #X_Predict = [X_Predict]
        X_Predict_Final = np.array(X_Predict_Final)
        print("Predict: ", NN.predict(X_Predict_Final))
        print("Storing model and weights")
        NN.save_model(json_filename, h5_filename)
        print("Done!")


class ProcessTweetsGloveOnePass:
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
        with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter = ',')
            for r in csv_file:
                if i !=0:
                    tweet = r[0]
                    label = int(r[1])
                    if (label == 2):
                        label = 0
                    X_all.append(tweet)
                    Y_all.append(label)
                i = i + 1
        print("Data Ingested")
        # divide the data into training and test
        num_data = len(X_all)
        limit = math.ceil(num_data * 0.60)
        X_train_sentences = X_all
        Y_train = Y_all
        # divide the data into X_train, Y_train, X_test, Y_test
        #X_train_sentences = X_all[0: limit]
        #Y_train = Y_all[0: limit]
        #X_test_sentences = X_all[limit:]
        #Y_test = Y_all[limit:]
        #print("Data Divided")
        #Get embeeding
        G = GloveEmbedding(self.embedding_filename)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_train_indices, max_len  = S.map_sentence_list(X_train_sentences)
        print("Train data mappend to indices")
        P = PadSentences(max_len)
        X_train_pad = P.pad_list(X_train_indices)
        print("Train data padded")
        # Trim
        trim_size = 33
        max_len = trim_size
        Trim = TrimSentences(trim_size)
        X_train_pad = Trim.trim_list(X_train_pad)
        #convert to numPY arrays
        X_train = np.array(X_train_pad)
        Y_train = np.array(Y_train)
        print("Train data convert to numpy arrays")
        NN = TweetSentiment2LSTM2Dense(max_len, G)

        print("model created")
        NN.build(first_layer_units = max_len, second_layer_units = max_len, relu_dense_layer=16, dense_layer_units = 1,
                 first_layer_dropout=0, second_layer_dropout=0)
        print("model built")
        NN.summary()
        sgd = SGD(lr=0.01, momentum=0.9, decay=0.1, nesterov=False)
        rmsprop = RMSprop(decay=0.001)
        NN.compile(optimizer=rmsprop, loss="binary_crossentropy", metrics=['accuracy', precision, recall, f1, fprate])
        print("model compiled")
        print("Begin training")
        callback = TensorBoard(log_dir="/tmp/logs")
        class_weight = {1: (1-0.63), 0: 0.63}
        history = NN.fit(X_train, Y_train, epochs=epochs, callbacks=[callback], validation_split=0.2, class_weight = class_weight)
        print("Model trained")
        # X_test_indices, max_len = S.map_sentence_list(X_test_sentences)
        # print("Test data mapped")
        # X_test_pad = P.pad_list(X_test_indices)
        # print("Test data padded")
        # X_test = np.array(X_test_pad)
        # Y_test = np.array(Y_test)
        # print("Test data converted to numpy arrays")
        # loss, acc = NN.evaluate(X_test, Y_test, callbacks=[callback])
        # print("accuracy: ", acc)
        T = "I have a bad case of vomit"
        X_Predict = ["my zika is bad", "i love colombia every day of the week dude", "my has been tested for ebola", "there is a diarrhea outbreak in the city"]
        X_Predict_Idx, max_len2 = S.map_sentence_list(X_Predict)
        i =0
        for s in X_Predict_Idx:
            print(str(i)+ ": ", s)
            i = i + 1
        print(X_Predict)
        X_Predict_Final = P.pad_list(X_Predict_Idx)
        X_Predict_Final = Trim.trim_list(X_Predict_Final)

        #X_Predict = [X_Predict]
        X_Predict_Final = np.array(X_Predict_Final)
        print("Predict: ", NN.predict(X_Predict_Final))
        print("Storing model and weights")
        NN.save_model(json_filename, h5_filename)
        if plot:
            print("Ploting")
            self.plot(history)
        print("Done!")


class ProcessTweetsGloveOnePassSM:
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
        with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter = ',')
            for r in csv_file:
                if i !=0:
                    tweet = r[0]
                    label = r[1]
                    X_all.append(tweet)
                    Y_all.append(label)
                i = i + 1
        print("Data Ingested")
        # divide the data into training and test
        num_data = len(X_all)
        limit = math.ceil(num_data * 0.60)
        X_train_sentences = X_all
        Y_train = Y_all
        # divide the data into X_train, Y_train, X_test, Y_test
        #X_train_sentences = X_all[0: limit]
        #Y_train = Y_all[0: limit]
        #X_test_sentences = X_all[limit:]
        #Y_test = Y_all[limit:]
        #print("Data Divided")
        #Get embeeding
        G = GloveEmbedding(self.embedding_filename )
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_train_indices, max_len  = S.map_sentence_list(X_train_sentences)
        print("Train data mappend to indices")
        P = PadSentences(max_len)
        X_train_pad = P.pad_list(X_train_indices)
        print("Train data padded")
        #convert to numPY arrays
        X_train = np.array(X_train_pad)
        Y_train = np.array(Y_train)
        Y_train = to_categorical(Y_train, num_classes=3)
        print("Train data convert to numpy arrays")
        NN = TweetSentiment2LSTM2DenseSM(max_len, G)
        print("model created")
        kernel_regularizer = l2(0.001)
        kernel_regularizer = None
        NN.build(first_layer_units = max_len, second_layer_units=max_len, relu_dense_layer=5, dense_layer_units=3, first_layer_dropout=0.3, second_layer_dropout=0.6, l2 = kernel_regularizer)
        print("model built")
        NN.summary()
        sgd = SGD(lr=0.001, momentum=0.09, decay=0.001, nesterov=True)
        rmsprop = RMSprop(decay=0.003)
        adam = Adam(lr=0.1, decay=0.05)
        NN.compile(optimizer=rmsprop, loss="categorical_crossentropy", metrics=['accuracy', precision, recall, f1, fprate])
        print("model compiled")
        print("Begin training")
        callback = TensorBoard(log_dir="/tmp/logs")
        w_dict = {0: 0.31, 1: 0.63, 2 : 0.06}
        history = NN.fit(X_train, Y_train, epochs=epochs, callbacks=[callback], validation_split=0.2, class_weight= w_dict)
        print("Model trained")
        # X_test_indices, max_len = S.map_sentence_list(X_test_sentences)
        # print("Test data mapped")
        # X_test_pad = P.pad_list(X_test_indices)
        # print("Test data padded")
        # X_test = np.array(X_test_pad)
        # Y_test = np.array(Y_test)
        # print("Test data converted to numpy arrays")
        # loss, acc = NN.evaluate(X_test, Y_test, callbacks=[callback])
        # print("accuracy: ", acc)
        T = "I have a bad case of vomit"
        X_Predict = ["my zika is bad", "i love colombia", "my has been tested for ebola", "there is a diarrhea outbreak in the city"]
        X_Predict_Idx, max_len2 = S.map_sentence_list(X_Predict)
        i =0
        for s in X_Predict_Idx:
            print(str(i)+ ": ", s)
            i = i + 1
        print(X_Predict)
        X_Predict_Final = P.pad_list(X_Predict_Idx)
        #X_Predict = [X_Predict]
        X_Predict_Final = np.array(X_Predict_Final)
        print("Predict: ", NN.predict(X_Predict_Final))
        print("Storing model and weights")
        NN.save_model(json_filename, h5_filename)
        if plot:
            print("Ploting")
            self.plot(history)
        print("Done!")


class ProcessTweetsWord2VecOnePassSM:
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

    def process(self, json_filename, h5_filename, plot=False, epochs = 100, vect_dimensions = 100):
        np.random.seed(11)
        # open the file with tweets
        X_all = []
        Y_all = []
        with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter = ',')
            for r in csv_file:
                if i !=0:
                    tweet = r[0]
                    label = r[1]
                    X_all.append(tweet)
                    Y_all.append(label)
                i = i + 1
        print("Data Ingested")
        # divide the data into training and test
        num_data = len(X_all)
        limit = math.ceil(num_data * 0.60)
        X_train_sentences = X_all
        Y_train = Y_all
        # divide the data into X_train, Y_train, X_test, Y_test
        #X_train_sentences = X_all[0: limit]
        #Y_train = Y_all[0: limit]
        #X_test_sentences = X_all[limit:]
        #Y_test = Y_all[limit:]
        #print("Data Divided")
        #Get embeeding
        G = Word2VecEmbedding(self.embedding_filename, dimensions=vect_dimensions)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_train_indices, max_len  = S.map_sentence_list(X_train_sentences)
        print("Train data mappend to indices")
        P = PadSentences(max_len)
        X_train_pad = P.pad_list(X_train_indices)
        print("Train data padded")
        trim_size = 40
        Trim = TrimSentences(trim_size)
        X_train_pad = Trim.trim_list(X_train_pad)
        #convert to numPY arrays
        X_train = np.array(X_train_pad)
        Y_train = np.array(Y_train)
        Y_train = to_categorical(Y_train, num_classes=3)
        print("Train data convert to numpy arrays")
        #NN = TweetSentiment2LSTM2DenseSM(max_len, G)
        NN = TweetSentiment2LSTM2DenseSM(trim_size, G)

        #print("Build GRU")
        #NN = TweetSentimentGRUSM(max_len, G)

        print("model created")
        kernel_regularizer = l2(0.001)
        #kernel_regularizer = None
        NN.build(first_layer_units = 100, second_layer_units=100, relu_dense_layer=100, dense_layer_units=3, first_layer_dropout=0.1, second_layer_dropout=0.1, l2 = None)
        print("model built")
        NN.summary()
        sgd = SGD(lr=0.01, momentum=0.009, decay=0.001, nesterov=True)
        rmsprop = RMSprop(decay=0.005)
        adam = Adam(lr=0.1, decay=0.05)
        NN.compile(optimizer=rmsprop, loss="categorical_crossentropy", metrics=['accuracy'])
        print("model compiled")
        print("Begin training")
        callback = TensorBoard(log_dir="/tmp/logs")
        history = NN.fit(X_train, Y_train, epochs=epochs, callbacks=[callback], validation_split=0.3 )
        print("Model trained")
        # X_test_indices, max_len = S.map_sentence_list(X_test_sentences)
        # print("Test data mapped")
        # X_test_pad = P.pad_list(X_test_indices)
        # print("Test data padded")
        # X_test = np.array(X_test_pad)
        # Y_test = np.array(Y_test)
        # print("Test data converted to numpy arrays")
        # loss, acc = NN.evaluate(X_test, Y_test, callbacks=[callback])
        # print("accuracy: ", acc)
        T = "I have a bad case of vomit"
        X_Predict = ["my zika is bad", "i love colombia", "my has been tested for ebola", "there is a diarrhea outbreak in the city"]
        X_Predict_Idx, max_len2 = S.map_sentence_list(X_Predict)
        i =0
        for s in X_Predict_Idx:
            print(str(i)+ ": ", s)
            i = i + 1
        print(X_Predict)
        X_Predict_Final = P.pad_list(X_Predict_Idx)
        X_Predict_Final = Trim.trim_list(X_Predict_Final)
        #X_Predict = [X_Predict]
        X_Predict_Final = np.array(X_Predict_Final)
        print("Predict: ", NN.predict(X_Predict_Final))
        print("Storing model and weights")
        NN.save_model(json_filename, h5_filename)
        if plot:
            print("Ploting")
            self.plot(history)
        print("Done!")



class ProcessTweetsWord2VecOnePassCNN:
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

    def process(self, json_filename, h5_filename, plot=False, epochs = 100, vect_dimensions = 100):
        np.random.seed(11)
        # open the file with tweets
        X_all = []
        Y_all = []
        with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter = ',')
            for r in csv_file:
                if i !=0:
                    tweet = r[0]
                    label = r[1]
                    X_all.append(tweet)
                    Y_all.append(label)
                i = i + 1
        print("Data Ingested")
        # divide the data into training and test
        num_data = len(X_all)
        limit = math.ceil(num_data * 0.60)
        X_train_sentences = X_all
        Y_train = Y_all
        # divide the data into X_train, Y_train, X_test, Y_test
        #X_train_sentences = X_all[0: limit]
        #Y_train = Y_all[0: limit]
        #X_test_sentences = X_all[limit:]
        #Y_test = Y_all[limit:]
        #print("Data Divided")
        #Get embeeding
        G = Word2VecEmbedding(self.embedding_filename, dimensions=vect_dimensions)
        #G = GloveEmbedding(self.embedding_filename, dimensions=50)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_train_indices, max_len  = S.map_sentence_list(X_train_sentences)
        print("Train data mappend to indices")
        P = PadSentences(max_len)
        X_train_pad = P.pad_list(X_train_indices)
        print("Train data padded")
        print("X[0], ", X_train_pad[0])
        #convert to numPY arrays
        X_train = np.array(X_train_pad)
        Y_train = np.array(Y_train)
        Y_train = to_categorical(Y_train, num_classes=3)
        print("Train data convert to numpy arrays")
        NN = TweetSentimentCNN(max_len, G)
        #print("Build GRU")
        #NN = TweetSentimentGRUSM(max_len, G)

        print("model created")
        kernel_regularizer = l2(0.001)
        #kernel_regularizer = None
        NN.build(filters=10, kernel_size=3, first_dropout=0.2, second_dropout=0.2, padding='same')
        print("model built")
        NN.summary()
        sgd = SGD(lr=0.03, momentum=0.009, decay=0.001, nesterov=True)
        rmsprop = RMSprop(decay=0.003)
        adam = Adam(lr=0.1, decay=0.05)
        NN.compile(optimizer=rmsprop, loss="categorical_crossentropy", metrics=['accuracy'])
        print("model compiled")
        print("Begin training")
        callback = TensorBoard(log_dir="/tmp/logs")
        history = NN.fit(X_train, Y_train, epochs=epochs, batch_size=64, callbacks=[callback], validation_split=0.3 )
        print("Model trained")
        # X_test_indices, max_len = S.map_sentence_list(X_test_sentences)
        # print("Test data mapped")
        # X_test_pad = P.pad_list(X_test_indices)
        # print("Test data padded")
        # X_test = np.array(X_test_pad)
        # Y_test = np.array(Y_test)
        # print("Test data converted to numpy arrays")
        # loss, acc = NN.evaluate(X_test, Y_test, callbacks=[callback])
        # print("accuracy: ", acc)
        T = "I have a bad case of vomit"
        X_Predict = ["my zika is bad", "i love colombia", "my has been tested for ebola", "there is a diarrhea outbreak in the city"]
        X_Predict_Idx, max_len2 = S.map_sentence_list(X_Predict)
        i =0
        for s in X_Predict_Idx:
            print(str(i)+ ": ", s)
            i = i + 1
        print(X_Predict)
        X_Predict_Final = P.pad_list(X_Predict_Idx)
        #X_Predict = [X_Predict]
        X_Predict_Final = np.array(X_Predict_Final)
        print("Predict: ", NN.predict(X_Predict_Final))
        print("Storing model and weights")
        NN.save_model(json_filename, h5_filename)
        if plot:
            print("Ploting")
            self.plot(history)
        print("Done!")


class ProcessTweetsWord2VecOnePass2DCNN:
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

    def process(self, json_filename, h5_filename, plot=False, epochs = 100, vect_dimensions = 100):
        np.random.seed(11)
        # open the file with tweets
        X_all = []
        Y_all = []
        with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter = ',')
            for r in csv_file:
                if i !=0:
                    tweet = r[0]
                    label = r[1]
                    X_all.append(tweet)
                    Y_all.append(label)
                i = i + 1
        print("Data Ingested")
        # divide the data into training and test
        num_data = len(X_all)
        limit = math.ceil(num_data * 0.60)
        X_train_sentences = X_all
        Y_train = Y_all
        # divide the data into X_train, Y_train, X_test, Y_test
        #X_train_sentences = X_all[0: limit]
        #Y_train = Y_all[0: limit]
        #X_test_sentences = X_all[limit:]
        #Y_test = Y_all[limit:]
        #print("Data Divided")
        #Get embeeding
        G = Word2VecEmbedding(self.embedding_filename, dimensions=vect_dimensions)
        #G = GloveEmbedding(self.embedding_filename, dimensions=50)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_train_indices, max_len  = S.map_sentence_list(X_train_sentences)
        print("Train data mappend to indices")
        if max_len % 2 !=0:
            max_len = max_len + 1

        P = PadSentences(max_len)
        X_train_pad = P.pad_list(X_train_indices)
        print("Train data padded")
        trim_size = 40
        Trim = TrimSentences(trim_size)
        X_train_pad = Trim.trim_list(X_train_pad)
        print("X[0], ", X_train_pad[0])
        #convert to numPY arrays
        X_train = np.array(X_train_pad)
        Y_train = np.array(Y_train)
        Y_train = to_categorical(Y_train, num_classes=3)
        print("Train data convert to numpy arrays")
        NN = TweetSentiment2DCNN(trim_size, G)
        #print("Build GRU")
        #NN = TweetSentimentGRUSM(max_len, G)

        print("model created")
        kernel_regularizer = l2(0.001)
        #kernel_regularizer = None
        NN.build(filters=3, kernel_size=3, first_dropout=0.15, second_dropout=0.15, padding='valid')
        print("model built")
        NN.summary()
        sgd = SGD(lr=0.03, momentum=0.009, decay=0.001, nesterov=True)
        rmsprop = RMSprop(decay=0.003)
        adam = Adam(lr=0.1, decay=0.05)
        sgd = SGD(lr=0.05)
        NN.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
        print("model compiled")
        print("Begin training")
        callback = TensorBoard(log_dir="/tmp/logs")
        class_weight = {0: 5., 1: 10., 2: 1.}
        history = NN.fit(X_train, Y_train, epochs=epochs, batch_size=32, callbacks=[callback], validation_split=0.2, class_weight=class_weight )
        print("Model trained")
        # X_test_indices, max_len = S.map_sentence_list(X_test_sentences)
        # print("Test data mapped")
        # X_test_pad = P.pad_list(X_test_indices)
        # print("Test data padded")
        # X_test = np.array(X_test_pad)
        # Y_test = np.array(Y_test)
        # print("Test data converted to numpy arrays")
        # loss, acc = NN.evaluate(X_test, Y_test, callbacks=[callback])
        # print("accuracy: ", acc)
        T = "I have a bad case of vomit"
        X_Predict = ["my zika is bad", "i love colombia", "my has been tested for ebola", "there is a diarrhea outbreak in the city"]
        X_Predict_Idx, max_len2 = S.map_sentence_list(X_Predict)
        i =0
        for s in X_Predict_Idx:
            print(str(i)+ ": ", s)
            i = i + 1
        print(X_Predict)
        X_Predict_Final = P.pad_list(X_Predict_Idx)
        X_Predict_Final = Trim.trim_list(X_Predict_Final)
        #X_Predict = [X_Predict]
        X_Predict_Final = np.array(X_Predict_Final)
        print("Predict: ", NN.predict(X_Predict_Final))
        print("Storing model and weights")
        NN.save_model(json_filename, h5_filename)
        if plot:
            print("Ploting")
            self.plot(history)
        print("Done!")

class ProcessTweetsWord2VecOnePass2DCNNv2:
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

    def process(self, json_filename, h5_filename, plot=False, epochs = 100, vect_dimensions = 100):
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
                    # tweet = r[0]
                    # label = r[1]
                    # X_all.append(tweet)
                    # Y_all.append(label)
                i = i + 1
        np.random.shuffle(All)

        for r in All:
            tweet = r[0]
            label = r[1]
            X_all.append(tweet)
            Y_all.append(label)
        print("Data Ingested")
        # divide the data into training and test
        num_data = len(X_all)
        limit = math.ceil(num_data * 0.60)
        X_train_sentences = X_all
        Y_train = Y_all
        # divide the data into X_train, Y_train, X_test, Y_test
        #X_train_sentences = X_all[0: limit]
        #Y_train = Y_all[0: limit]
        #X_test_sentences = X_all[limit:]
        #Y_test = Y_all[limit:]
        #print("Data Divided")
        #Get embeeding
        G = Word2VecEmbedding(self.embedding_filename, dimensions=vect_dimensions)
        #G = GloveEmbedding(self.embedding_filename, dimensions=50)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_train_indices, max_len  = S.map_sentence_list(X_train_sentences)
        print("Train data mappend to indices")
        if max_len % 2 !=0:
            max_len = max_len + 1

        P = PadSentences(max_len)
        X_train_pad = P.pad_list(X_train_indices)
        print("Train data padded")
        # TRIM
        trim_size = 40
        Trim = TrimSentences(trim_size)
        X_train_pad = Trim.trim_list(X_train_pad)
        print("X[0], ", X_train_pad[0])
        #convert to numPY arrays
        X_train = np.array(X_train_pad)
        Y_train = np.array(Y_train)
        Y_train = to_categorical(Y_train, num_classes=3)
        print("Train data convert to numpy arrays")
        #NN = TweetSentiment2DCNN(trim_size, G)
        NN = TweetSentiment2DCNNv2(trim_size, G)

        #print("Build GRU")
        #NN = TweetSentimentGRUSM(max_len, G)

        print("model created")
        kernel_regularizer = l2(0.001)
        #kernel_regularizer = None
        NN.build(filters=3, first_dropout=0.01, second_dropout=0.01, padding='valid', dense_units=16)
        print("model built")
        NN.summary()
        sgd = SGD(lr=0.03, momentum=0.009, decay=0.001, nesterov=True)
        rmsprop = RMSprop(decay=0.003)
        adam = Adam(lr=0.1, decay=0.05)
        sgd = SGD(lr=0.05)
        NN.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
        print("model compiled")
        print("Begin training")
        callback = TensorBoard(log_dir="/tmp/logs")
        history = NN.fit(X_train, Y_train, epochs=epochs, batch_size=32, callbacks=[callback], validation_split=0.4 )
        print("Model trained")
        # X_test_indices, max_len = S.map_sentence_list(X_test_sentences)
        # print("Test data mapped")
        # X_test_pad = P.pad_list(X_test_indices)
        # print("Test data padded")
        # X_test = np.array(X_test_pad)
        # Y_test = np.array(Y_test)
        # print("Test data converted to numpy arrays")
        # loss, acc = NN.evaluate(X_test, Y_test, callbacks=[callback])
        # print("accuracy: ", acc)
        T = "I have a bad case of vomit"
        X_Predict = ["my zika is bad", "i love colombia", "my has been tested for ebola", "there is a diarrhea outbreak in the city"]
        X_Predict_Idx, max_len2 = S.map_sentence_list(X_Predict)
        i =0
        for s in X_Predict_Idx:
            print(str(i)+ ": ", s)
            i = i + 1
        print(X_Predict)
        X_Predict_Final = P.pad_list(X_Predict_Idx)
        X_Predict_Final = Trim.trim_list(X_Predict_Final)
        #X_Predict = [X_Predict]
        X_Predict_Final = np.array(X_Predict_Final)
        print("Predict: ", NN.predict(X_Predict_Final))
        print("Storing model and weights")
        NN.save_model(json_filename, h5_filename)
        if plot:
            print("Ploting")
            self.plot(history)
        print("Done!")

class ProcessTweetsWord2VecOnePass2DCNNv4:
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

    def process(self, json_filename, h5_filename, plot=False, epochs = 100, vect_dimensions = 100):
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
                    # tweet = r[0]
                    # label = r[1]
                    # X_all.append(tweet)
                    # Y_all.append(label)
                i = i + 1
        np.random.shuffle(All)

        for r in All:
            tweet = r[0]
            label = r[1]
            if int(label) == 2:
                label = '0'
            X_all.append(tweet)
            Y_all.append(label)
        print("Data Ingested")
        # divide the data into training and test
        num_data = len(X_all)
        limit = math.ceil(num_data * 0.60)
        X_train_sentences = X_all
        Y_train = Y_all
        # divide the data into X_train, Y_train, X_test, Y_test
        #X_train_sentences = X_all[0: limit]
        #Y_train = Y_all[0: limit]
        #X_test_sentences = X_all[limit:]
        #Y_test = Y_all[limit:]
        #print("Data Divided")
        #Get embeeding
        G = Word2VecEmbedding(self.embedding_filename, dimensions=vect_dimensions)
        #G = GloveEmbedding(self.embedding_filename, dimensions=50)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_train_indices, max_len  = S.map_sentence_list(X_train_sentences)
        print("Train data mappend to indices")
        if max_len % 2 !=0:
            max_len = max_len + 1

        P = PadSentences(max_len)
        X_train_pad = P.pad_list(X_train_indices)
        print("Train data padded")
        # TRIM
        trim_size = 40
        Trim = TrimSentences(trim_size)
        X_train_pad = Trim.trim_list(X_train_pad)
        print("X[0], ", X_train_pad[0])
        #convert to numPY arrays
        X_train = np.array(X_train_pad)
        Y_train = np.array(Y_train)
        #Y_train = to_categorical(Y_train, num_classes=3)
        print("Train data convert to numpy arrays")
        #NN = TweetSentiment2DCNN(trim_size, G)
        NN = TweetSentiment2DCNNv4(trim_size, G)

        #print("Build GRU")
        #NN = TweetSentimentGRUSM(max_len, G)

        print("model created")
        kernel_regularizer = l2(0.001)
        #kernel_regularizer = None
        NN.build(filters=3, first_dropout=0.01, second_dropout=0.01, padding='valid', dense_units=16)
        print("model built")
        NN.summary()
        sgd = SGD(lr=0.03, momentum=0.009, decay=0.001, nesterov=True)
        rmsprop = RMSprop(decay=0.003)
        adam = Adam(lr=0.1, decay=0.05)
        sgd = SGD(lr=0.05)
        NN.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy', f1, precision, recall])
        print("model compiled")
        print("Begin training")
        callback = TensorBoard(log_dir="/tmp/logs")
        history = NN.fit(X_train, Y_train, epochs=epochs, batch_size=32, callbacks=[callback], validation_split=0.4 )
        print("Model trained")
        # X_test_indices, max_len = S.map_sentence_list(X_test_sentences)
        # print("Test data mapped")
        # X_test_pad = P.pad_list(X_test_indices)
        # print("Test data padded")
        # X_test = np.array(X_test_pad)
        # Y_test = np.array(Y_test)
        # print("Test data converted to numpy arrays")
        # loss, acc = NN.evaluate(X_test, Y_test, callbacks=[callback])
        # print("accuracy: ", acc)
        T = "I have a bad case of vomit"
        X_Predict = ["my zika is bad", "i love colombia", "my has been tested for ebola", "there is a diarrhea outbreak in the city"]
        X_Predict_Idx, max_len2 = S.map_sentence_list(X_Predict)
        i =0
        for s in X_Predict_Idx:
            print(str(i)+ ": ", s)
            i = i + 1
        print(X_Predict)
        X_Predict_Final = P.pad_list(X_Predict_Idx)
        X_Predict_Final = Trim.trim_list(X_Predict_Final)
        #X_Predict = [X_Predict]
        X_Predict_Final = np.array(X_Predict_Final)
        print("Predict: ", NN.predict(X_Predict_Final))
        print("Storing model and weights")
        NN.save_model(json_filename, h5_filename)
        if plot:
            print("Ploting")
            self.plot(history)
        print("Done!")


class ProcessTweetsWord2VecOnePassSMv2:
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

    def process(self, json_filename, h5_filename, plot=False, epochs = 100, vect_dimensions = 100):
        np.random.seed(11)
        # open the file with tweets
        X_all = []
        Y_all = []
        All = []
        with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter = ',')

            for r in csv_file:
                if i !=0:
                    All.append(r)
                    # tweet = r[0]
                    # label = r[1]
                    # X_all.append(tweet)
                    # Y_all.append(label)
                i = i + 1

        np.random.shuffle(All)
        for r in All:
            tweet = r[0]
            label = int(r[1])
            if label == 2:
                label = 0

            X_all.append(tweet)
            Y_all.append(label)

        print("Data Ingested")
        # divide the data into training and test
        num_data = len(X_all)
        limit = math.ceil(num_data * 0.60)
        X_train_sentences = X_all
        Y_train = Y_all
        # divide the data into X_train, Y_train, X_test, Y_test
        #X_train_sentences = X_all[0: limit]
        #Y_train = Y_all[0: limit]
        #X_test_sentences = X_all[limit:]
        #Y_test = Y_all[limit:]
        #print("Data Divided")
        #Get embeeding
        G = Word2VecEmbedding(self.embedding_filename, dimensions=vect_dimensions)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_train_indices, max_len  = S.map_sentence_list(X_train_sentences)
        print("Train data mappend to indices")
        P = PadSentences(max_len)
        X_train_pad = P.pad_list(X_train_indices)
        print("Train data padded")
        trim_size = 40
        Trim = TrimSentences(trim_size)
        X_train_pad = Trim.trim_list(X_train_pad)
        #convert to numPY arrays
        X_train = np.array(X_train_pad)
        Y_train = np.array(Y_train)
        #Y_train = to_categorical(Y_train, num_classes=3)
        print("Train data convert to numpy arrays")
        #NN = TweetSentiment2LSTM2DenseSM(max_len, G)
        NN = TweetSentiment2LSTM2DenseSMv2(trim_size, G)

        #print("Build GRU")
        #NN = TweetSentimentGRUSM(max_len, G)

        print("model created")
        kernel_regularizer = l2(0.001)
        #kernel_regularizer = None
        NN.build(first_layer_units = 100, second_layer_units=100, relu_dense_layer=100, dense_layer_units=3, first_layer_dropout=0.01, second_layer_dropout=0.01, l2 = None)
        print("model built")
        NN.summary()
        sgd = SGD(lr=0.01, momentum=0.009, decay=0.001, nesterov=True)
        rmsprop = RMSprop(decay=0.004)
        adam = Adam(lr=0.1, decay=0.05)
        NN.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy', precision, recall, f1, fprate])
        print("model compiled")
        print("Begin training")
        callback = TensorBoard(log_dir="/tmp/logs")
        class_weight = {1: (1 - 0.63), 0: 0.63}
        history = NN.fit(X_train, Y_train, epochs=epochs, callbacks=[callback], validation_split=0.4, class_weight= class_weight )
        print("Model trained")
        # X_test_indices, max_len = S.map_sentence_list(X_test_sentences)
        # print("Test data mapped")
        # X_test_pad = P.pad_list(X_test_indices)
        # print("Test data padded")
        # X_test = np.array(X_test_pad)
        # Y_test = np.array(Y_test)
        # print("Test data converted to numpy arrays")
        # loss, acc = NN.evaluate(X_test, Y_test, callbacks=[callback])
        # print("accuracy: ", acc)
        T = "I have a bad case of vomit"
        X_Predict = ["my zika is bad", "i love colombia", "my has been tested for ebola", "there is a diarrhea outbreak in the city"]
        X_Predict_Idx, max_len2 = S.map_sentence_list(X_Predict)
        i =0
        for s in X_Predict_Idx:
            print(str(i)+ ": ", s)
            i = i + 1
        print(X_Predict)
        X_Predict_Final = P.pad_list(X_Predict_Idx)
        X_Predict_Final = Trim.trim_list(X_Predict_Final)
        #X_Predict = [X_Predict]
        X_Predict_Final = np.array(X_Predict_Final)
        print("Predict: ", NN.predict(X_Predict_Final))
        print("Storing model and weights")
        NN.save_model(json_filename, h5_filename)
        if plot:
            print("Ploting")
            self.plot(history)
        print("Done!")