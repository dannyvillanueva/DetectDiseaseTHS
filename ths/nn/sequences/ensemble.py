import numpy as np
np.random.seed(7)
from ths.utils.sentences import SentenceToEmbedding

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Concatenate, Bidirectional, BatchNormalization, GRU, Conv1D, Conv2D, MaxPooling2D, Flatten, Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.regularizers import l2


class TweetEnsemble1_2Layers:
    def __init__(self, max_sentence_len, embedding_builder):
        self.max_sentence_len = max_sentence_len
        self.embedding_builder = embedding_builder
        self.model = None
        self.sentiment_map = {0 : 'negative', 1 : 'positive', 2: 'neutral'}

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, relu_dense_layer = 32, dense_layer_units = 1):

        # Input Layer
        sentence_input = Input(shape = (self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        X1  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1a')(embeddings)
        X2  = LSTM(first_layer_units, return_sequences=False, name='LSTM_1b')(embeddings)

        #X  = Bidirectional(LSTM(first_layer_units, return_sequences=True, name='LSTM_1'))(embeddings)
        # Dropout regularization
        X1 = Dropout(first_layer_dropout, name="DROPOUT_1") (X1)
        # Second LSTM Layer
       # X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2", kernel_regularizer=l2(0.1))(X)
        # Second Layer Dropout
        X1  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2")(X1)
        X1 = Dropout(second_layer_dropout, name="DROPOUT_2")(X1)

        X = Concatenate()([X1, X2])
        # Send to a Dense Layer with softmax activation
        X= Dense(256, activation='relu', name="DENSE_1")(X)
        X= Dense(128, activation='relu', name="DENSE_2")(X)

        X = Dense(dense_layer_units,name="DENSE_Final")(X)
        X = Activation("sigmoid", name="SIGMOID_1")(X)
        # create the model
        self.model  = Model(input = sentence_input, output=X)

    def summary(self):
        self.model.summary()

    def pretrained_embedding_layer(self):
        # create Keras embedding layer
        word_to_idx, idx_to_word, word_embeddings = self.embedding_builder.read_embedding()
        #vocabulary_len = len(word_to_idx) + 1
        vocabulary_len = len(word_to_idx)
        emb_dimension = self.embedding_builder.get_dimensions()
        # get the matrix for the sentences
        embedding_matrix = word_embeddings
        #embedding_matrix = np.vstack([word_embeddings, np.zeros((vocabulary_len,))])

        # embedding layer
        embedding_layer = Embedding(input_dim=vocabulary_len, output_dim=emb_dimension, trainable=False, name="EMBEDDING")
        embedding_layer.build((None,))
        embedding_layer.set_weights([embedding_matrix])
        return embedding_layer

    def compile(self, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, X, Y, epochs = 50, batch_size = 32, shuffle=True, callbacks=None, validation_split=0.0, class_weight=None):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                       validation_split=validation_split, class_weight=class_weight)

    def evaluate(self, X_test, Y_test):
        return self.model.evaluate(X_test, Y_test)

    def predict(self, X):
        return self.model.predict(X)

    def get_sentiment(self, prediction):
        return np.argmax(prediction)

    def sentiment_string(self, sentiment):
        return self.sentiment_map[sentiment]

    def save_model(self, json_filename, h5_filename):
        json_model = self.model.to_json()
        with open(json_filename, "w") as json_file:
            json_file.write(json_model)
        self.model.save_weights(h5_filename)
        return

class TweetEnsemble1_3Layers(TweetEnsemble1_2Layers):
    def __init__(self, max_sentence_len, embedding_builder):
        self.max_sentence_len = max_sentence_len
        self.embedding_builder = embedding_builder
        self.model = None
        self.sentiment_map = {0 : 'negative', 1 : 'positive', 2: 'neutral'}

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, relu_dense_layer = 32, dense_layer_units = 1):

        # Input Layer
        sentence_input = Input(shape = (self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        X1  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1a')(embeddings)
        X2  = LSTM(first_layer_units, return_sequences=False, name='LSTM_1b')(embeddings)

        #X  = Bidirectional(LSTM(first_layer_units, return_sequences=True, name='LSTM_1'))(embeddings)
        # Dropout regularization
        X1 = Dropout(first_layer_dropout, name="DROPOUT_1") (X1)
        # Second LSTM Layer
       # X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2", kernel_regularizer=l2(0.1))(X)
        # Second Layer Dropout
        #third layer
        X3 = LSTM(second_layer_units, return_sequences=True, name="LSTM_2b")(X1)
        X3 = Dropout(second_layer_dropout, name="DROPOUT_3")(X3)
        X3 = LSTM(second_layer_units, return_sequences=False, name="LSTM_3")(X3)
        X1  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2a")(X1)
        X1 = Dropout(second_layer_dropout, name="DROPOUT_2")(X1)

        X = Concatenate()([X1, X2, X3])
        # Send to a Dense Layer with softmax activation
        X= Dense(256, activation='relu', name="DENSE_1")(X)
        X= Dense(128, activation='relu', name="DENSE_2")(X)

        X = Dense(dense_layer_units,name="DENSE_Final")(X)
        X = Activation("sigmoid", name="SIGMOID_1")(X)
        # create the model
        self.model  = Model(input = sentence_input, output=X)
