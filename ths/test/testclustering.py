from ths.utils.files import GloveEmbedding
from ths.utils.sentences import SentenceToEmbeddingWithEPSILON
import csv
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as metrics_nmi, adjusted_rand_score as metrics_ari
from keras.optimizers import SGD
from keras.initializers import VarianceScaling
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Flatten, Reshape, GRU
from keras.engine.topology import Layer, InputSpec
from keras.utils import plot_model
import keras.backend as K
import pandas
from ths.utils.contractions import expandContractions


def get_glove(glove_dims): # get glove embedding matrix
    if glove_dims == 50:
        G = GloveEmbedding(filename="../test/data/glove.twitter.27B.50d.txt", dimensions=50)
    elif glove_dims==200:
        G = GloveEmbedding(filename="../test/data/glove.twitter.27B.200d.txt", dimensions=200)
    elif glove_dims==300:
        G = GloveEmbedding(filename="../test/data/glove.840B.300d.txt", dimensions=300)
    else:
        print("Wrong Number of dimensions")
        exit(0)
    word_to_idx, idx_to_word, embedding = G.read_embedding()
    #S = SentenceToIndices(word_to_idx)
    SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
    return SE


def get_max_len(file, SE):
    max_len = 0
    for line in file:
        matrix = SE.map_sentence(line.lower())
        tweet_len = len(matrix)
        if tweet_len > max_len:
            max_len = tweet_len
    return max_len

def fix_text_format(d):
    a = []
    # remove white spaces and expand contractions
    for row in d:
        line = row.strip()  # remove white spaces
        line = expandContractions(line)  # expand contractions
        line = line.replace("“", '').replace("”", '').replace('"', '')  # remove quotation marks
        a.append(line)
    return a

def read_file(size, labeled):
    data = []
    if labeled:
        y = []
        try:
            datafile = open("data/data" + size + "_labeled.csv", "r")
            with datafile as f:
                for row in csv.reader(f):
                    newline = " ".join(row[0].split())
                    newline = newline.replace("’", "'")
                    data.append(newline)
                    y.append(int(row[1]))
        except Exception as e:
            print(e)
        data = fix_text_format(data)
        return data, np.array(y)
    else:
        try:
            datafile = open("data/data" + size + ".txt", "r", encoding='utf-8')
            with datafile as f:
                for line in f:
                    newline = " ".join(line.split())
                    data.append(newline)
        except Exception as e:
            print(e)
        data = fix_text_format(data)
        return data


def get_embedding_data(SE, size, labeled):
    dict_1  = {}
    finaldata = []
    if labeled:
        data, y = read_file(size, labeled)
        max_len = get_max_len(data, SE)
        for line in data:
            emb = SE.map_sentence(line.lower(), max_len=max_len)
            finaldata.append(emb)
            dict_1[line] = emb
        return finaldata, max_len, dict_1, y
    else:
        data = read_file(size, labeled)
        max_len = get_max_len(data, SE)
        for line in data:
            emb = SE.map_sentence(line.lower(), max_len=max_len)
            finaldata.append(emb)
            dict_1[line] = emb
        return finaldata, max_len, dict_1


def get_dict_1(file, size, labeled):
    data = []
    dict_1 = {}
    count = 0
    if labeled:
        data, y = read_file(size, labeled)
    else:
        data = read_file(size, labeled)
    for line in data:
        dict_1[line] = file[count]
        count += 1
    if labeled:
        return dict_1, y
    else:
        return dict_1


def metrics_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def baseline_clustering_accuracy(file_2d, n_clusters, y):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    y_pred_kmeans = kmeans.fit_predict(file_2d)
    accuracy = metrics_acc(np.array(y), y_pred_kmeans)
    return kmeans, y_pred_kmeans, accuracy


def autoencoder(type, n_features, max_len, init):
    if type == "dense_t1":
        # define Input
        input = Input(shape=(max_len, n_features), name="Input")
        # encode layers
        X = Flatten(name='Flatten_encoder')(input)
        X = Dense(128, name='Dense_128_encoder')(X)
        X = Dense(256, name='Dense_256_encoder')(X)
        # final encode layer (code)
        encoder = Dense(128, name='final_encoder')(X)
        # decode layers
        X = Dense(256, name='Dense_256_decoder')(encoder)
        X = Dense(128, name='Dense_128_decoder')(X)
        X = Dense(max_len * n_features, name='Dense_Flatten_decoder')(X)
        auto_encoder = Reshape((max_len, n_features), name='Reshape_decoder')(X)
        # output model
        autoencoder_model = Model(input, auto_encoder, name="auto_encoder")
        encoder_model = Model(input, encoder, name="encoder")
        # return all models
        return autoencoder_model, encoder_model
    if type == "dense_t2":
        # define Input
        input = Input(shape=(max_len * n_features, ), name="Input")
        # encode layers
        X = Dense(128, kernel_initializer=init, name='Dense_128_encoder')(input)
        X = Dense(256, kernel_initializer=init, name='Dense_256_encoder')(X)
        # final encode layer (code)
        encoder = Dense(128, kernel_initializer=init, name='final_encoder')(X)
        # decode layers
        X = Dense(256, kernel_initializer=init, name='Dense_256_decoder')(encoder)
        X = Dense(128, kernel_initializer=init, name='Dense_128_decoder')(X)
        auto_encoder = Dense(max_len * n_features, kernel_initializer=init, name='Dense_Flatten_decoder')(X)
        # output model
        autoencoder_model = Model(input, auto_encoder, name="auto_encoder")
        encoder_model = Model(input, encoder, name="encoder")
        # return all models
        return autoencoder_model, encoder_model
    if type == "full_lstm":
        # define input
        input = Input(shape=(max_len, n_features))
        # encode layers
        encoder = LSTM(128, return_sequences=True, return_state=True)
        encoder_outputs, state_h, state_c = encoder(input)
        # final encode layer (code)
        encoder_states = [state_h, state_c]
        encoder_states = [state_h]
        # decode layers
        auto_encoder = LSTM(n_features, return_sequences=True)(encoder_outputs)
        autoencoder_model = Model(input, auto_encoder)
        encoder_model = Model(input, encoder_states)
        # return all models
        return autoencoder_model, encoder_model

    if type == "full_gru":
        # define input
        input = Input(shape=(max_len, n_features))
        # encode layers
        encoder = GRU(128, return_sequences=True, return_state=True, activation='tanh', \
                  recurrent_activation='hard_sigmoid', use_bias=True, \
                  kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', \
                  bias_initializer='zeros')
        encoder_outputs, state_h = encoder(input)
        # final encode layer (code)
        encoder_states = [state_h]
        # decode layers
        auto_encoder = GRU(n_features, return_sequences=True)(encoder_outputs)
        autoencoder_model = Model(input, auto_encoder)
        encoder_model = Model(input, encoder_states)
        return autoencoder_model, encoder_model

    if type == "gru_lstm":
        # define input
        input = Input(shape=(max_len, n_features))
        # encode layers
        encoder = GRU(128, return_sequences=True, name = "encoder_gru_1")
        encoder_outputs, state_h = encoder(input)
         # final encode layer (code)
        encoder_states = [state_h]
        # decode layers
        auto_encoder = LSTM(n_features, return_sequences=True)(encoder_outputs)
        encoder_model = Model(input, encoder_states)
        autoencoder_model = Model(input, auto_encoder)
        return autoencoder_model, encoder_model
    print("Wrong model type! ")


def kernel_initializer():
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    return init


def pretrain_optimizer(opt):
    if opt == 'sgd':
        return SGD(lr=1, momentum=0.9)
    if opt == 'adam':
        return 'adam'
    print("Wrong pre train optimizer name")


def train_optimizer(opt):
    if opt == 'sgd':
        return SGD(lr=0.01, momentum=0.9)
    if opt == 'adam':
        return 'adam'
    print("Wrong train optimizer name")


def plot_model_loss(h):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_model_keras(m, ae, e):
    plot_model(ae, to_file="../test/output/ae_" + m + ".png", show_shapes=True)
    plot_model(e, to_file="../test/output/e_" + m + ".png", show_shapes=True)


class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, **kwargs):
        self.output_dim = n_clusters
        self.initial_weights = weights
        super(ClusteringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) == 2: # Dense and GRU is 2D
            input_dim = input_shape[1]
        if len(input_shape) == 3: # lstm code is 3D
            pass
        self.clusters = self.add_weight((self.output_dim, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(ClusteringLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        print("input shape: ", inputs.shape)

        return [[0.2, 0.1, 0.7],[0.6, 0.2, 0.2]] # list of ndarrays of (3,)

    def compute_output_shape(self, input_shape):
        assert input_shape
        return input_shape[0], self.output_dim


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


if __name__ == "__main__":

    # load data
    glove_dims = 50 # options: 50, 200
    labeled_data = 0  # options: 0, 1
    compute_baseline_kmeans = 0 # 0, 1
    stored_embedding = 0  # options: 0, 1
    load_autoencoder = 0
    load_final_model = 0
    SE = get_glove(glove_dims)
    y = None  #
    if labeled_data:
        if stored_embedding:
            file_size = ('15k', 72)  # options: ('15k', 72)
            outfile = '../test/output/embedded_' + file_size[0] + '_' + str(glove_dims) + 'd.npy'
            file = np.load(outfile)
            max_len = file_size[1]
            dict_1, y = get_dict_1(file, file_size[0], labeled_data)
        else:
            size = '15k' # size of raw_tweet csv file: data15k_labeled.csv
            file, max_len, dict_1, y = get_embedding_data(SE, size, labeled_data)
            print("Max Len: ", max_len)
            outfile = '../test/output/embedded_' + size + '_'+ str(glove_dims) + 'd.npy'
            np.save(outfile, file)
    else:
        if stored_embedding:
            file_size = ('100k', 46)  # options: ('20k', ), ('100k', 46), ('500k', )
            outfile = '../test/output/embedded_'+file_size[0]+'_' + str(glove_dims) + 'd.npy'
            file = np.load(outfile)
            max_len = file_size[1]
            dict_1 =get_dict_1(file, file_size[0], labeled_data)
        else:
            size = '100k' # size of raw_tweet txt file: data100k.txt
            file, max_len, dict_1 = get_embedding_data(SE, size, labeled_data)
            print("Max Len: ", max_len)
            outfile = '../test/output/embedded_' + size + '_' + str(glove_dims) + 'd.npy'
            np.save(outfile, file)

    # setting baseline clustering
    file = np.array(file)
    n_clusters = 3
    file_2d = file.reshape((file.shape[0], -1))
    if compute_baseline_kmeans:
        kmeans, y_pred_kmeans, accuracy = baseline_clustering_accuracy(file_2d, n_clusters, y)
        print("Baseline K-Means Accuracy: ", accuracy)

    # setting auto encoder hyper parameters
    n_features = glove_dims
    epochs = 20
    batch_size = 256
    kernel_init = kernel_initializer()
    optimizer = pretrain_optimizer(opt="adam")  # sgd, adam
    loss = "mse"  # mse,
    val_split = 0.15

    # training auto encoder model
    save_dir = '../test/output'
    model_name = 'dense_t1'  # dense_t1, dense_t2, full_lstm, full_gru, gru_lstm
    x_input = None
    if model_name == 'dense_t1':
        x_input = file
    if model_name == 'dense_t2':
        x_input = file_2d
    if model_name == 'full_lstm':
        x_input = file
    if model_name == 'full_gru':
        x_input = file
    auto_encoder, encoder = autoencoder(model_name, n_features, max_len, init=kernel_init) #, init=kernel_init)
    if not load_autoencoder:
        auto_encoder.compile(optimizer=optimizer, loss=loss, metrics=['mse'])
        hist = auto_encoder.fit(x_input, x_input, batch_size=batch_size, epochs=epochs, validation_split=val_split)

        # plot model and save model weights
        plot_model_keras(model_name, auto_encoder, encoder)
        plot_model_loss(hist)
        auto_encoder.save_weights(save_dir + "/ae_weights_" + model_name + "_" + str(glove_dims) + "d.h5")
        encoder.save_weights(save_dir + "/e_weights_" + model_name + "_" + str(glove_dims) + "d.h5")

    # load pretrain autoencoder weights
    auto_encoder.load_weights(save_dir + "/ae_weights_" + model_name + "_" + str(glove_dims) + "d.h5")

    # save final encoded data
    encoded_data = encoder.predict(x_input)
    outfile_encoder = '../test/output/encoded_data_' + model_name + "_" + str(glove_dims) + "d.npy"
    np.save(outfile_encoder, np.array(encoded_data))
    """
    # build clustering model
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    clustering_model = Model(inputs=encoder.input, outputs=clustering_layer)
    
    if not load_final_model:
        plot_model(clustering_model, to_file="../test/output/cluster_" + model_name + ".png", show_shapes=True)
        clustering_model.compile(optimizer=train_optimizer("sgd"), loss='kld')
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(encoder.predict(x_input))
        clustering_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        y_pred_last = np.copy(y_pred)
        loss = 0
        index = 0
        max_iter = 8000
        update_interval = 256
        index_array = np.arange(x_input.shape[0])
        tol = 0.001  # tolerance threshold to stop training

        # training final model
        for ite in range(max_iter):
            if ite % update_interval == 0:
                q = clustering_model.predict(x_input, verbose=1)
                # update the auxiliary target distribution p
                #p = target_distribution(q)
                y_pred = q.argmax(1)
                if y is not None: # evaluate the clustering performance
                    acc = np.round(metrics_acc(y, y_pred), 3)
                    nmi = np.round(metrics_nmi(y, y_pred), 3)
                    ari = np.round(metrics_ari(y, y_pred), 3)
                    loss = np.round(loss, 3)
                    print('Iter %d: acc = %.3f, nmi = %.3f, ari = %.3f' % (ite, acc, nmi, ari), ' ; loss=', loss)
            idx = index_array[index: min((index + 1), x_input.shape[0])]
            loss = clustering_model.train_on_batch(x = x_input[idx], y = y[idx])
            index = index + 1 if (index + 1) * 1 <= x_input.shape[0] else 0

        # save final model weights
        clustering_model.save_weights(save_dir + "/cluster_model_final_" + model_name + ".h5")

    # load final model weights
    clustering_model.load_weights(save_dir + "/cluster_model_final_" + model_name + ".h5")
    
    # final evaluation
    q = clustering_model.predict(x_input, verbose=1)
    p = target_distribution(q)  # update the auxiliary target distribution p

    # evaluate the clustering performance
    y_pred = q.argmax(1)
    if y is not None:
        acc = np.round(metrics_acc(y, y_pred), 3)
        nmi = np.round(metrics_nmi(y, y_pred), 3)
        ari = np.round(metrics_ari(y, y_pred), 3)
        loss = np.round(loss, 3)
        print('Acc = %.3f, nmi = %.3f, ari = %.3f' % (acc, nmi, ari), ' ; loss=', loss)
    """