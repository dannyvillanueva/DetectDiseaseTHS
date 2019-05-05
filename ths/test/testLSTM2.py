from random import randint
import matplotlib.pyplot as plt
from numpy import argmax
from numpy import array_equal
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Flatten, Reshape, GRU
from ths.utils.files import GloveEmbedding
from ths.utils.sentences import SentenceToIndices, SentenceToEmbeddingWithEPSILON
import numpy as np
from keras.optimizers import SGD
from sklearn.cluster import KMeans

def get_max_len(file, SE):
    max_len = 0
    #c = 0
    for line in file:
        matrix = SE.map_sentence(line.lower())
     #   print("line ",c, ' :',line.lower())
        tweet_len = len(matrix)
        if tweet_len > max_len:
            max_len = tweet_len
      #  c+=1
    return max_len

def getGlove():
    G = GloveEmbedding("../test/data/glove.twitter.27B.50d.txt")
    word_to_idx, idx_to_word, embedding = G.read_embedding()
    S = SentenceToIndices(word_to_idx)
    SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
    return SE

def getUnderscoreGlove():
    # test_ = SE.map_sentence("_", max_len= max_len)
    # print(test_)
    shape = (50,)
    underscore_vector = np.ones(shape) * -2
    return underscore_vector

def get_data():
    SE = getGlove()
    data = []
    dictionary1  = {}
    dictionary2  = {}
    try:
        datafile = open("data/data100k.txt", "r", encoding='utf-8')
        with datafile as f:
            for line in f:
                newline = " ".join(line.split())
                data.append(newline)
    except Exception as e:
        print(e)

    max_len = get_max_len(data, SE)
    finaldata = []
    for line in data:
        emb = SE.map_sentence(line.lower(), max_len=max_len)
        finaldata.append(emb)
        dictionary1[line] = emb

    return finaldata, max_len

# prepare data for the LSTM
def get_dataset(n_in, n_out, file):
    X1, X2, y = list(), list(), list()
    for source in file:
        target = source
        X1.append(source)
        X2.append(target)

    X1 = np.array(X1)
    X2 = np.array(X2)

    return X1, X2

# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model

def define_models_lstm(n_input, n_output, n_units, max_len):
    # define training encoder
    encoder_inputs = Input(shape=(max_len, n_input))
    encoder = LSTM(n_units, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states)
    # define training decoder
    decoder_lstm = LSTM(n_output, return_sequences=True)(encoder_outputs)
    autoencoder_model = Model(encoder_inputs, decoder_lstm)
    # return all models
    return encoder_model, autoencoder_model

def define_models_gru(n_input, n_output, n_units, max_len):
    # define training encoder
    encoder_inputs = Input(shape=(max_len, n_input))
    encoder = GRU(n_units, return_sequences=True, return_state=True, activation='tanh', \
                  recurrent_activation='hard_sigmoid', use_bias=True, \
                  kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', \
                  bias_initializer='zeros')

    encoder_outputs, state_h = encoder(encoder_inputs)
    encoder_states = [state_h]
    encoder_model = Model(encoder_inputs, encoder_states)

    # define training decoder
    decoder_lstm = GRU(n_output, return_sequences=True)(encoder_outputs)
    autoencoder_model = Model(encoder_inputs, decoder_lstm)

    # return all models
    return encoder_model, autoencoder_model

def define_models_dense1(n_input, max_len):
    # define Input
    input = Input(shape=(max_len, n_input))
    # define encode layers
    X = Flatten(name='Flatten_encoder')(input)
    X = Dense(128, name='Dense_128_encoder')(X)
    X = Dense(256, name='Dense_256_encoder')(X)
    encoded = Dense(128, name='final_encoder')(X)
    # define decode layers
    X = Dense(256, name='Dense_256_decoder')(encoded)
    X = Dense(128, name='Dense_128_decoder')(X)
    X = Dense(max_len * n_input, name='Dense_Flatten_decoder')(X)
    X = Reshape((max_len, n_input), name='Reshape_decoder')(X)
    decoded = X
    # output
    autoencoder_model = Model(input, decoded, name="auto_encoder")
    encoder_model = Model(input, encoded, name="encoder")
    # return all models
    return autoencoder_model, encoder_model

def define_models_dense2(n_input, max_len):
    # define Input
    input = Input(shape=(max_len * n_input, ))
    # define encode layers
    X = Dense(128, name='Dense_128_encoder')(input)
    X = Dense(256, name='Dense_256_encoder')(X)
    encoded = Dense(128, name='final_encoder')(X)
    # define decode layers
    X = Dense(256, name='Dense_256_decoder')(encoded)
    X = Dense(128, name='Dense_128_decoder')(X)
    X = Dense(max_len * n_input, name='Dense_Flatten_decoder')(X)
    decoded = X
    # output
    autoencoder_model = Model(input, decoded, name="auto_encoder")
    encoder_model = Model(input, encoded, name="encoder")
    # return all models
    return autoencoder_model, encoder_model

# main
if __name__ == "__main__":
    SE = getGlove()
    embedding = 1
    outfile = '../test/output/embedded_15k.npy'

    if embedding:
        #max_len for 100k:46, 15k:72
        file = np.load(outfile)
        max_len = 72
    else:
        file, max_len = get_data()
        print("Max_len: ", max_len)
        np.save(outfile, file)

    n_features = 50
    n_steps_in = max_len
    n_steps_out = max_len
    n_clusters = 3
    model = 'dense1'
"""
    x_file = file.reshape((file.shape[0], -1))
    print("File shape: ", file.shape)
    print("XFile shape: ", x_file.shape)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
    print("Start Base line K-Means clustering accuracy")
    y_pred_kmeans = kmeans.fit_predict(x_file)
    print("End Base line K-Means clustering accuracy")
    print("Predict: \n", y_pred_kmeans[0:10])
"""

if model == 'dense1':
    output_model, code = define_models_dense1(n_features, max_len)
    code.summary()
    output_model.compile(optimizer="adam", loss='mse', metrics=['mse'])
    X1, X2 = get_dataset(n_steps_in, n_steps_out, file)
    #print(X1.shape, X2.shape, y.shape)
    hist = output_model.fit(X1, X1, epochs=30, batch_size= 64, validation_split = 0.15)

    print("loss: \n", hist.history['loss'])
    print("loss: \n", hist.history['val_loss'])

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if model == 'dense2':
    file_2d = file.reshape((file.shape[0], -1))
    output_model, code = define_models_dense2(n_features, max_len)
    output_model.summary()
    code.summary()
    output_model.compile(optimizer="adam", loss='mse', metrics=['mse'])
    #print(file_2d.shape, file_2d.shape)
    hist = output_model.fit(file_2d , file_2d, epochs=30, batch_size= 64, validation_split= 0.15)
    print("loss: \n", hist.history['loss'])
    print("loss: \n", hist.history['val_loss'])

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# define model for LSTM
if model == 'lstm':
    encoder, autoencoder, = define_models_lstm(n_features, n_features, 128, max_len)
    encoder.summary()
    autoencoder.summary()
    #pretrain_optimizer = SGD(lr=0.5, momentum=0.9)
    autoencoder.compile(optimizer="adam", loss='mse', metrics=['mse'])
    #autoencoder.compile(optimizer=pretrain_optimizer, loss='mse', metrics=['mse'])
    #autoencoder.compile(optimizer="rmsprop", loss='mse', metrics=['mse'])
    # generate training dataset
    X1, X2 = get_dataset(n_steps_in, n_steps_out, file)
    print(X1.shape, X2.shape)
    # train model
    autoencoder.fit(X1, X1, epochs=20)

if model == 'gru':
    encoder, autoencoder, = define_models_gru(n_features, n_features, 128, max_len)
    encoder.summary()
    autoencoder.summary()
    autoencoder.compile(optimizer="adam", loss='mse', metrics=['mse'])
    X1, X2 = get_dataset(n_steps_in, n_steps_out, file)
    print(X1.shape, X2.shape)
    # train model
    autoencoder.fit(X1, X1, epochs=20, batch_size= 256, validation_split = 0.15)
