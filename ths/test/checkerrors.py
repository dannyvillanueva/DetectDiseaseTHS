from ths.nn.sequences.tweets import TweetSentiment2DCNNv2_1
from ths.utils.files import GloveEmbedding, Word2VecEmbedding
from ths.utils.sentences import SentenceToIndices, SentenceToEmbedding, PadSentences, TrimSentences
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.regularizers import l2
from ths.nn.metrics.f1score import f1, precision, recall, fprate
from keras.models import model_from_json

import numpy as np
import csv
import math

np.random.seed(11)


def main(model_file, model_weights, labeled_tweets, embedding_filename):
    # load json and create model
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights)
    print("Loaded model from disk")
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # open the file with tweets
    X_all = []
    Y_all = []
    All  = []

    with open(labeled_tweets, "r", encoding="ISO-8859-1") as f:
        i = 0
        csv_file = csv.reader(f, delimiter = ',')
        ones_count = 0

        for r in csv_file:
            if i !=0:
                label = int(r[1])
                if (label == 1) or (label == 2):
                    if ones_count <= 13000:
                        All.append(r)
                        ones_count += 1
                else :
                    All.append(r)
                # tweet = r[0]
                # label = r[1]
                # X_all.append(tweet)
                # Y_all.append(label)
            i = i + 1

    ones_count = 0
    for r in All:
        tweet = r[0]
        label = int(r[1])
        if (label == 2):
            label = 0
        # if (label == 1) and (ones_count <= 4611):
        #     X_all.append(tweet)
        #     Y_all.append(label)
        #     ones_count +=1
        # elif (label == 0):
        X_all.append(tweet)
        Y_all.append(label)

    print("Data Ingested")
    # divide the data into training and test
    num_data = len(X_all)
    limit = math.ceil(num_data * 0.60)
    X_train_sentences = X_all
    Y_train = Y_all
    # divide the data into X_train, Y_train, X_test, Y_test
    # X_train_sentences = X_all[0: limit]
    # Y_train = Y_all[0: limit]
    # X_test_sentences = X_all[limit:]
    # Y_test = Y_all[limit:]
    # print("Data Divided")
    # Get embeeding
    # G = Word2VecEmbedding(self.embedding_filename, dimensions=vect_dimensions)
    G = GloveEmbedding(embedding_filename, dimensions=50)
    word_to_idx, idx_to_word, embedding = G.read_embedding()
    S = SentenceToIndices(word_to_idx)
    X_train_indices, max_len = S.map_sentence_list(X_train_sentences)
    print("Train data mappend to indices")
    if max_len % 2 != 0:
        max_len = max_len + 1

    P = PadSentences(max_len)
    X_train_pad = P.pad_list(X_train_indices)
    print("Train data padded")
    # TRIM
    trim_size = max_len
    Trim = TrimSentences(trim_size)
    X_train_pad = Trim.trim_list(X_train_pad)
    print("X[0], ", X_train_pad[0])
    # convert to numPY arrays
    X_train = np.array(X_train_pad)
    Y_train = np.array(Y_train)
    ones_count = np.count_nonzero(Y_train)
    zeros_count = len(Y_train) - ones_count
    print("ones count: ", ones_count)
    print("zeros count: ", zeros_count)
    # Y_train = to_categorical(Y_train, num_classes=3)
    print("Train data convert to numpy arrays")
    Preds = loaded_model.predict(X_train)
    Preds = ((Preds >= 0.5)*1).flatten()
    with open("data/errors.csv", "w") as f:
        csv_writer = csv.writer(f, delimiter = ",")
        i = 0
        err_count = 0
        for r in All:
            tweet = r[0]
            label = int(r[1])
            if label == 2:
                label = 0
            if Preds[i] != label:
                err_count += 1
                error_pred = []
                error_pred.append(tweet)
                error_pred.append(label)
                error_pred.append(Preds[i])
                csv_writer.writerow(error_pred)
            i += 1
        print("Error count: ", err_count)


    # Build model from JSON and h5 file
    # Test all mislabels

# joderme
if __name__ == "__main__":
    labeled_tweets = "data/cleantextlabels4.csv"
    embeddings_filename = "data/glove.6B.50d.txt"
    model_file_base = "trained/model2dcnn2"
    model_file = model_file_base + ".json"
    model_weights = model_file_base + ".h5"
    main(model_file=model_file, model_weights= model_weights, labeled_tweets=labeled_tweets, embedding_filename=embeddings_filename)