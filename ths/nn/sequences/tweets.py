import numpy as np
np.random.seed(7)
from ths.utils.sentences import SentenceToEmbedding

from keras.models import Model
from keras.layers import TimeDistributed, Dense, Input, Dropout, LSTM, Activation, Bidirectional, BatchNormalization, GRU, Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Flatten, Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, Permute, Multiply, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.regularizers import l2

#np.random.seed(1)

class TweetSentiment2LSTM:

    def __init__(self, max_sentence_len, embedding_builder):
        self.max_sentence_len = max_sentence_len
        self.embedding_builder = embedding_builder
        self.model = None
        self.sentiment_map = {0 : 'negative', 1 : 'positive', 2: 'neutral'}

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, dense_layer_units = 2):

        # Input Layer
        sentence_input = Input(shape = (self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        #X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(embeddings)
        X  = Bidirectional(LSTM(first_layer_units, return_sequences=True, name='LSTM_1'))(embeddings)

        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_1") (X)
        # Second LSTM Layer
       # X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2", kernel_regularizer=l2(0.1))(X)
        # Second Layer Dropout
        X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2")(X)
        X = Dropout(second_layer_dropout, name="DROPOUT_2")(X)
        # Send to a Dense Layer with softmax activation
        X = Dense(dense_layer_units,name="DENSE_1")(X)
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

class TweetSentiment3LSTM:

    def __init__(self, max_sentence_len, embedding_builder):
        self.max_sentence_len = max_sentence_len
        self.embedding_builder = embedding_builder
        self.model = None
        self.sentiment_map = {0 : 'negative', 1 : 'positive', 2: 'neutral'}

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, third_layer_units = 128, third_layer_dropout = 0.5, dense_layer_units = 2):

        # Input Layer
        sentence_input = Input(shape = (self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(embeddings)
        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_1") (X)
        # Second LSTM Layer
        X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_2')(X)
        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_2") (X)
        # Third layer
        X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_3")(X)
        # Third Layer Dropout
        X = Dropout(second_layer_dropout, name="DROPOUT_3")(X)
        # Send to a Dense Layer with softmax activation
        X = Dense(dense_layer_units,name="DENSE_1")(X)
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

    def fit(self, X, Y, epochs = 50, batch_size = 32, shuffle=True, validation_split=0.0):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, validation_split=validation_split)

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


class TweetSentiment2LSTM2Dense(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, third_layer_units = 128, third_layer_dropout = 0.5,
              relu_dense_layer = 64, dense_layer_units = 2):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        # X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(embeddings)
        X = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(embeddings)

        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_1")(X)
        # Second LSTM Layer
        # X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2", kernel_regularizer=l2(0.1))(X)
        # Second Layer Dropout
        X = LSTM(256, return_sequences=False, name="LSTM_2")(X)
        X = Dropout(second_layer_dropout, name="DROPOUT_2")(X)
        #X = Dense(128, name="DENSE_1", activation='relu')(X)
        # Send to a Dense Layer with sigmoid activation
        X = Dense(dense_layer_units, name="DENSE_2")(X)
        #X = Activation("sigmoid", name="SIGMOID_1")(X)
        X = Activation("softmax", name="SIGMOID_1")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)

    def fit(self, X, Y, epochs = 50, batch_size = 32, shuffle=True, callbacks=None, validation_split=0.0, class_weight=None):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                       validation_split=validation_split, class_weight=class_weight, verbose=2)

class TweetSentiment2LSTM2DenseSM(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, third_layer_units = 128, third_layer_dropout = 0.5,
              relu_dense_layer = 64, dense_layer_units = 2, l2 = None):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        # X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(embeddings)
        #X = Bidirectional(LSTM(first_layer_units, return_sequences=True, name='LSTM_1'), name="DB_LSTM1")(embeddings)
        X = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(embeddings)

        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_1")(X)
        # Second LSTM Layer
        # X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2", kernel_regularizer=l2(0.1))(X)
        # Second Layer Dropout
        X = LSTM(second_layer_units, return_sequences=True, name="LSTM_2")(X)
        X = Dropout(second_layer_dropout, name="DROPOUT_2")(X)

        X = LSTM(second_layer_units, return_sequences=False, name="LSTM_3")(X)
        X = Dropout(second_layer_dropout, name="DROPOUT_3")(X)

        X = Dense(100, name="DENSE_1", activation='relu', kernel_regularizer=l2)(X)
        X = BatchNormalization()(X)
        X = Dense(64, name="DENSE_2", activation='relu', kernel_regularizer=l2)(X)
        X = BatchNormalization()(X)
        X = Dense(32, name="DENSE_3", activation='relu', kernel_regularizer=l2)(X)
        X = BatchNormalization()(X)
        # Send to a Dense Layer with sigmoid activation
        X = Dense(dense_layer_units, name="DENSE_4", kernel_regularizer=l2)(X)
        X = BatchNormalization()(X)
        X = Activation("softmax", name="softmax")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)

    def fit(self, X, Y, epochs = 50, batch_size = 32, shuffle=True, callbacks=None, validation_split=0.0, class_weight=None):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                       validation_split=validation_split, class_weight=class_weight)

class TweetSentimentGRUSM(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, third_layer_units = 128, third_layer_dropout = 0.5,
              relu_dense_layer = 64, dense_layer_units = 2, l2 = None):
        l2 = None
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        # X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(embeddings)
        #X = Bidirectional(GRU(first_layer_units, return_sequences=False, name='GRU'), name="BD_GRU")(embeddings)

        X = GRU(first_layer_units, return_sequences=False, name='GRU') (embeddings)

        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_1")(X)
        # Second LSTM Layer
        # X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2", kernel_regularizer=l2(0.1))(X)
        # Second Layer Dropout
        #X = LSTM(second_layer_units, return_sequences=False, name="LSTM_2", kernel_regularizer=l2)(X)
        #X = Dropout(second_layer_dropout, name="DROPOUT_2")(X)
        X = Dense(4*relu_dense_layer, name="DENSE_1", activation='tanh', kernel_regularizer=l2)(X)
        #X = BatchNormalization()(X)
        X = Dense(relu_dense_layer, name="DENSE_2", activation='tanh', kernel_regularizer=l2)(X)
        #X = BatchNormalization()(X)
        # Send to a Dense Layer with sigmoid activation
        X = Dense(dense_layer_units, name="DENSE_3", kernel_regularizer=l2)(X)
        #X = BatchNormalization()(X)
        X = Activation("softmax", name="softmax")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)


class TweetSentimentCNN:
    def __init__(self, max_sentence_len, embedding_builder):
        self.max_sentence_len = max_sentence_len
        self.embedding_builder = embedding_builder
        self.model = None

    def build(self, first_dropout=0.0, padding='same', filters=250, kernel_size=5, strides=1, activation='relu',
              dense_units=64, second_dropout=0.0):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        #embeddings = Embedding(5000,
        #                       self.embedding_builder.get_dimensions(),input_length= self.max_sentence_len )(sentence_input)

        print("embedding: ", embeddings)
        # First Dropout
        #X = Dropout(first_dropout, name="DROPOUT_1")(embeddings)
        #print("embeddings.shape: ", embeddings.shape)


        X = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings)

        # 1D Convolutional Neural Network
        X = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation,
                   name="CONV1D_1")(embeddings)
        X = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation,
                   name="CONV1D_2")(X)
        X = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation,
                   name="CONV1D_3")(X)
        X = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation,
                   name="CONV1D_4")(X)
        # 1D Pooling layer
        #X = GlobalAveragePooling1D(name="GLOBALMAXPOOLING1D_1")(X)
        #X =
        #X = AveragePooling1D(name="GLOBALMAXPOOLING1D_1")(X)

        #X = MaxPooling1D(name="GLOBALMAXPOOLING1D_1")(X)

        #X = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation,
        #           name="CONV1D_2")(X)
        #X = AveragePooling1D(name="GLOBALMAXPOOLING1D_2")(X)
        # Dense Layers
        X = Flatten()(X)

        X = Dense(units=2*dense_units, activation='relu', name="DENSE_1")(X)

        X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)

        X = Dropout(second_dropout, name="DROPOUT_2")(X)
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_3")(X)
        X = Dropout(second_dropout, name="DROPOUT_3")(X)

        # Final layer
        X = Dense(3, activation= "softmax")(X)
        # create the model

        self.model = Model(input=sentence_input, output=X)

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

    def summary(self):
        self.model.summary()

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

    #def sentiment_string(self, sentiment):
    #    return self.sentiment_map[sentiment]

    def save_model(self, json_filename, h5_filename):
        json_model = self.model.to_json()
        with open(json_filename, "w") as json_file:
            json_file.write(json_model)
        self.model.save_weights(h5_filename)
        return


class TweetSentiment2DCNN:
    def __init__(self, max_sentence_len, embedding_builder):
        self.max_sentence_len = max_sentence_len
        self.embedding_builder = embedding_builder
        self.model = None

    def build(self, first_dropout=0.0, padding='same', filters=250, kernel_size=5, strides=(1,1), activation='relu',
              dense_units=128, second_dropout=0.0):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        #embeddings = Embedding(5000,
        #                       self.embedding_builder.get_dimensions(),input_length= self.max_sentence_len )(sentence_input)

        print("embedding: ", embeddings)
        # First Dropout
        #X = Dropout(first_dropout, name="DROPOUT_1")(embeddings)
        #print("embeddings.shape: ", embeddings.shape)
        # 1D Convolutional Neural Network

        X = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings)
        print("X", X)
        X = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1), padding=padding, activation=activation,
                   name="CONV2D_1")(X)
        print("Conv2D 1: ", X)
        # 1D Pooling layer
        pool_dim = int((self.max_sentence_len/2))
        pool_size = (2,2)
        print("pool_size: ", pool_size)
        X = MaxPooling2D(pool_size=pool_size, name="MAXPOOLIN2D_1")(X)
        print("Max Pool: ", X)
        X = Dropout(first_dropout, name="DROPOUT_1")(X)
        #X = AveragePooling1D(name="GLOBALMAXPOOLING1D_1")(X)
        #new_shape  = X.output_shape() + (1,)
        #print("new_shape: ", new_shape)
        #X = Reshape(new_shape)(X)

        X = Conv2D(filters=2*filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation,
                  name="CONV2D_2")(X)
        pool_dim = int(pool_dim/2)
        pool_size = (2, 2)
        X = MaxPooling2D(pool_size=pool_size, name="MAXPOOLIN2D_2")(X)
        #X = AveragePooling1D(name="GLOBALMAXPOOLING1D_2")(X)
        X = Dropout(first_dropout, name="DROPOUT_2")(X)

        X = Conv2D(filters=2*2*filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation,
                  name="CONV2D_3")(X)
        pool_dim = int(pool_dim/2)
        pool_size = (2, 2)
        X = MaxPooling2D(pool_size=pool_size, name="MAXPOOLIN2D_3")(X)
        #X = AveragePooling1D(name="GLOBALMAXPOOLING1D_2")(X)
        X = Dropout(first_dropout, name="DROPOUT_3")(X)


        # Dense Layers
        X = Flatten()(X)
        X = Dense(units=dense_units, activation='relu', name="DENSE_1")(X)

        X = Dropout(second_dropout, name="DROPOUT_4")(X)

        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_2")(X)
        X = Dropout(second_dropout, name="DROPOUT_5")(X)

        # Final layer
        X = Dense(3, activation= "softmax")(X)
        # create the model

        self.model = Model(input=sentence_input, output=X)

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

    def summary(self):
        self.model.summary()

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

    #def sentiment_string(self, sentiment):
    #    return self.sentiment_map[sentiment]

    def save_model(self, json_filename, h5_filename):
        json_model = self.model.to_json()
        with open(json_filename, "w") as json_file:
            json_file.write(json_model)
        self.model.save_weights(h5_filename)
        return


class  TweetSentiment2DCNNv2(TweetSentiment2DCNN):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_dropout=0.0, padding='same', filters=4, kernel_size=(1,1), strides=(1,1), activation='relu',
              dense_units=64, second_dropout=0.0):

        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")

        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)

        # Reshape with channels
        X = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings)

        # First convolutional layer
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 4
        kernel_size = (kernel_height, kernel_width)
        X = Conv2D(filters=3, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV2D_1")(X)
        #MAX pooling
        pool_height =  self.max_sentence_len - kernel_height + 1  # assumes zero padding and stride of 1
        pool_size = (pool_height, 1)
        X = MaxPooling2D(pool_size=pool_size, name = "MAXPOOL_1")(X)

        #Flatten
        X = Flatten()(X)

        # # First dense layer
        dense_units = 128
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_1")(X)
        X = Dropout(second_dropout, name="DROPOUT_1")(X)

        # # Second dense layer
        # X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)
        # X = Dropout(second_dropout, name="DROPOUT_2")(X)
        #
        # # Third layer
        # X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_3")(X)
        # X = Dropout(second_dropout, name="DROPOUT_3")(X)

        # Final layer
        X = Dense(3, activation= "softmax")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)

class  TweetSentiment2DCNNv3(TweetSentiment2DCNN):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_dropout=0.0, padding='same', filters=4, kernel_size=(1,1), strides=(1,1), activation='relu',
              dense_units=64, second_dropout=0.0):

        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")

        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)

        # Reshape with channels
        X = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings)

        # First convolutional layer
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 4
        kernel_size = (kernel_height, kernel_width)
        X = Conv2D(filters=3, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV2D_1")(X)
        #MAX pooling
        pool_height =  2  # assumes zero padding and stride of 1
        pool_size = (pool_height, 1)
        X = MaxPooling2D(pool_size=pool_size, name = "MAXPOOL_1")(X)

        # Second Conv layer
        kernel_size = (kernel_height, 1)
        X = Conv2D(filters=6, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV2D_2")(X)
        #MAX pooling
        pool_height =  int(np.floor((self.max_sentence_len - kernel_height + 1)/2))  # assumes zero padding and stride of 1
        pool_height = int(pool_height - kernel_height + 1)
        print("pool_height", pool_height)
        pool_size = (pool_height, 1)
        #print("pool_size: ", pool_size)
        X = MaxPooling2D(pool_size=pool_size, name = "MAXPOOL_2")(X)

        #Flatten
        X = Flatten()(X)

        # # First dense layer
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_1")(X)
        X = Dropout(second_dropout, name="DROPOUT_1")(X)
        #
        # # Second dense layer
        # X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)
        # X = Dropout(second_dropout, name="DROPOUT_2")(X)
        #
        # # Third layer
        # X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_3")(X)
        # X = Dropout(second_dropout, name="DROPOUT_3")(X)

        # Final layer
        X = Dense(3, activation= "softmax")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)


class TweetSentiment2DCNNv4(TweetSentiment2DCNN):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_dropout=0.0, padding='same', filters=4, kernel_size=(1, 1), strides=(1, 1), activation='relu',
              dense_units=64, second_dropout=0.0):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")

        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)

        # Reshape with channels
        X = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings)

        # First convolutional layer
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 4
        kernel_size = (kernel_height, kernel_width)
        X = Conv2D(filters=3, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV2D_1")(X)
        # MAX pooling
        pool_height = self.max_sentence_len - kernel_height + 1  # assumes zero padding and stride of 1
        pool_size = (pool_height, 1)
        X = MaxPooling2D(pool_size=pool_size, name="MAXPOOL_1")(X)

        # Flatten
        X = Flatten()(X)

        # # First dense layer
        dense_units = 128
        X = Dense(units=int(dense_units / 2), activation='relu', name="DENSE_1")(X)
        X = Dropout(second_dropout, name="DROPOUT_1")(X)

        # # Second dense layer
        # X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)
        # X = Dropout(second_dropout, name="DROPOUT_2")(X)
        #
        # # Third layer
        # X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_3")(X)
        # X = Dropout(second_dropout, name="DROPOUT_3")(X)

        # Final layer
        X = Dense(1, activation="sigmoid")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)

class TweetSentiment2LSTM2DenseSMv2(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, third_layer_units = 128, third_layer_dropout = 0.5,
              relu_dense_layer = 64, dense_layer_units = 2, l2 = None):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        # X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(embeddings)
        #X = Bidirectional(LSTM(first_layer_units, return_sequences=True, name='LSTM_1'), name="DB_LSTM1")(embeddings)
        X = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(embeddings)

        # Dropout regularization
        #X = Dropout(first_layer_dropout, name="DROPOUT_1")(X)
        # Second LSTM Layer
        # X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2", kernel_regularizer=l2(0.1))(X)
        # Second Layer Dropout
        #X = LSTM(second_layer_units, return_sequences=True, name="LSTM_2")(X)
        #X = Dropout(second_layer_dropout, name="DROPOUT_2")(X)

        X = LSTM(second_layer_units, return_sequences=False, name="LSTM_3")(X)
        X = Dropout(second_layer_dropout, name="DROPOUT_3")(X)

        # X = Dense(100, name="DENSE_1", activation='relu', kernel_regularizer=l2)(X)
        # X = BatchNormalization()(X)
        # X = Dense(64, name="DENSE_2", activation='relu', kernel_regularizer=l2)(X)
        # X = BatchNormalization()(X)
        # X = Dense(32, name="DENSE_3", activation='relu', kernel_regularizer=l2)(X)
        #X = BatchNormalization()(X)
        # Send to a Dense Layer with sigmoid activation
        X = Dense(1, name="DENSE_4", kernel_regularizer=l2)(X)
        #X = BatchNormalization()(X)
        X = Activation("sigmoid", name="sigmoid")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)

    def fit(self, X, Y, epochs = 50, batch_size = 32, shuffle=True, callbacks=None, validation_split=0.0, class_weight=None):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                       validation_split=validation_split, class_weight=class_weight)

class  TweetSentiment2DCNNv2_1(TweetSentiment2DCNN):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_dropout=0.0, padding='same', filters=4, kernel_size=(1,1), strides=(1,1), activation='relu',
              dense_units=64, second_dropout=0.0):

        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")

        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)

        # Reshape with channels
        X = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings)

        # First convolutional layer
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 3
        kernel_size = (kernel_height, kernel_width)
        X = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV2D_1")(X)
        #MAX pooling
        pool_height =  self.max_sentence_len - kernel_height + 1  # assumes zero padding and stride of 1
        pool_size = (pool_height, 1)
        X = MaxPooling2D(pool_size=pool_size, name = "MAXPOOL_1")(X)

        #Flatten
        X = Flatten()(X)

        # # First dense layer
        dense_units = 32
        #X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_1")(X)
        #X = Dropout(second_dropout, name="DROPOUT_1")(X)

        # # Second dense layer
        X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)
        X = Dropout(second_dropout, name="DROPOUT_2")(X)
        #
        # # Third layer
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_3")(X)
        X = Dropout(second_dropout, name="DROPOUT_3")(X)

        # Final layer
        X = Dense(1, activation= "sigmoid")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)

class TweetSentiment2LSTM2Dense3Layer(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)
    def get_model(self):
        return self.model

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, third_layer_units = 128, third_layer_dropout = 0.5,
              relu_dense_layer = 64, dense_layer_units = 2):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        # X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(embeddings)
        X = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(embeddings)

        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_1")(X)
        # Second LSTM Layer
        # X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2", kernel_regularizer=l2(0.1))(X)
        # Second Layer Dropout
        #X = LSTM(second_layer_units, return_sequences=True, name="LSTM_2")(X)
        #X = Dropout(second_layer_dropout, name="DROPOUT_2")(X)
        X = LSTM(second_layer_units, return_sequences=False, name="LSTM_3")(X)
        X = Dropout(third_layer_dropout, name="DROPOUT_3")(X)
        X = Dense(128, name="DENSE_1", activation='relu')(X)
        # Send to a Dense Layer with sigmoid activation
        X = Dense(dense_layer_units, name="DENSE_2")(X)
        X = Activation("sigmoid", name="SIGMOID_1")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)

    def fit(self, X, Y, epochs = 50, batch_size = 32, shuffle=True, callbacks=None, validation_split=0.0, class_weight=None, verbose=1):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                       validation_split=validation_split, class_weight=class_weight, verbose=verbose)

class TweetSentiment2LSTM2Dense4Layer(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, third_layer_units = 128, third_layer_dropout = 0.5,
              relu_dense_layer = 64, dense_layer_units = 2):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        # X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(embeddings)
        X = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(embeddings)

        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_1")(X)
        # Second LSTM Layer
        # X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2", kernel_regularizer=l2(0.1))(X)
        # Second Layer Dropout
        X = LSTM(second_layer_units, return_sequences=True, name="LSTM_2")(X)
        X = Dropout(second_layer_dropout, name="DROPOUT_2")(X)
        X = LSTM(second_layer_units, return_sequences=True, name="LSTM_3")(X)
        X = Dropout(third_layer_dropout, name="DROPOUT_3")(X)
        X = LSTM(second_layer_units, return_sequences=False, name="LSTM_4")(X)
        X = Dropout(third_layer_dropout, name="DROPOUT_4")(X)
        X = Dense(64, name="DENSE_1", activation='relu')(X)
        X = Dense(32, name="DENSE_2", activation='relu')(X)

        # Send to a Dense Layer with sigmoid activation
        X = Dense(dense_layer_units, name="Final_DENSE")(X)
        X = Activation("sigmoid", name="SIGMOID_1")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)

    def fit(self, X, Y, epochs = 50, batch_size = 32, shuffle=True, callbacks=None, validation_split=0.0, class_weight=None):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                       validation_split=validation_split, class_weight=class_weight)

class TweetSentiment2LSTM2Attention(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, third_layer_units = 128, third_layer_dropout = 0.5,
              relu_dense_layer = 64, dense_layer_units = 2):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        # X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(embeddings)
        X = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(embeddings)

        #Attention Layer
        attention = Permute((2, 1), name="Attention_Permute")(X)
        attention = Dense(self.max_sentence_len, activation='softmax', name="Attention_Dense")(attention)
        attention_probs = Permute((2, 1), name='attention_probs')(attention)
        output_attention_mul = Multiply(name='attention_multiplu')([X, attention_probs])

        # Final Dense Layer
        final_layer = Flatten()(output_attention_mul)
        final_layer = Dense(1, activation="sigmoid", name="Final_SIGMOID")(final_layer)

        self.model = Model(input=sentence_input, output=final_layer)

    def fit(self, X, Y, epochs = 50, batch_size = 32, shuffle=True, callbacks=None, validation_split=0.0, class_weight=None):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                       validation_split=validation_split, class_weight=class_weight)


class TweetSentiment2LSTM2Attentionv2(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, third_layer_units = 128, third_layer_dropout = 0.5,
              relu_dense_layer = 64, dense_layer_units = 2):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)

        X = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings)
        filters = 11
        kernel_size = 5
        padding = 'valid'
        activation = 'relu'
        # 1D Convolutional Neural Network
        X = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation,
                   name="CONV1D_1")(embeddings)
        # First LSTM Layer
        # X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(embeddings)
        #X = Bidirectional(LSTM(first_layer_units, return_sequences=True, name='LSTM_1'))(embeddings)
        #X = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(embeddings)
        X = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(X)

        X = Dropout(0, name="DROPOUT_1")(X)

        #X = LSTM(first_layer_units, return_sequences=True, name='LSTM_1a')(X)
        #X = Dropout(0.10, name="DROPOUT_1a")(X)
        #Attention Layer
        attention = Permute((2, 1), name="Attention_Permute")(X)
        #attention = Dense(self.max_sentence_len, activation='softmax', name="Attention_Dense")(attention)
        #attention = TimeDistributed(Dense(self.max_sentence_len, activation='softmax', name="Attention_Dense"))(attention)
        attention = TimeDistributed(Dense(68, activation='softmax', name="Attention_Dense"))(attention)

        attention_probs = Permute((2, 1), name='attention_probs')(attention)
        output_attention_mul = Multiply(name='attention_multiplu')([X, attention_probs])

        # second LSTM Layer
        X = LSTM(second_layer_units, return_sequences=False, name = 'LSTM_2')(output_attention_mul)
        X = Dropout(0, name="DROPOUT_2")(X)
        # Send to a Dense Layer with sigmoid activation
        X = Dense(32, name="DENSE_1")(X)

        X = Dense(dense_layer_units, name="DENSE_2")(X)
        X = Activation("softmax", name="SOFTMAX_1")(X)

        # Model
        self.model = Model(input=sentence_input, output=X)

    def fit(self, X, Y, epochs = 50, batch_size = 32, shuffle=True, callbacks=None, validation_split=0.0, class_weight=None):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                       validation_split=validation_split, class_weight=class_weight, verbose=2)



class TweetSentimentSeq2Seq(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, third_layer_units = 128, third_layer_dropout = 0.5,
              relu_dense_layer = 64, dense_layer_units = 2):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        # X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(embeddings)
        #X = LSTM(first_layer_units, return_sequences=False, name='LSTM_1')(embeddings)
        X = LSTM(128, return_sequences=False, name='LSTM_1')(embeddings)

        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_1")(X)

        # Reshape to make it 3D
        #X = Reshape((self.max_sentence_len, 1))(X)
        X = Reshape((128, 1))(X)

        # Second LSTM Layer
        #X  = LSTM(second_layer_units, return_sequences=True, name="LSTM_2")(X)
        X  = LSTM(128, return_sequences=True, name="LSTM_2")(X)

        # Second Layer Dropout
        #X = LSTM(256, return_sequences=False, name="LSTM_2")(X)
        #X = Dropout(second_layer_dropout, name="DROPOUT_2")(X)
        # X = Dense(128, name="DENSE_1", activation='relu')(X)
        # Send to a Dense Layer with sigmoid activation
        X = TimeDistributed(Dense(dense_layer_units, activation='relu', name="DENSE_1"))(X)
        X = Flatten()(X)
        #X = Dense(256, activation='relu', name="DENSE_1")(X)
        X = Dense(128, activation='relu', name="DENSE_2")(X)
        X = Dense(64, activation='relu', name="DENSE_3")(X)

        # X = Activation("sigmoid", name="SIGMOID_1")(X)
        X = Dense(3, name="FINAL_DENSE")(X)
        X = Activation("softmax", name="SOFTMAX_1")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)

    def fit(self, X, Y, epochs=50, batch_size=32, shuffle=True, callbacks=None, validation_split=0.0,
            class_weight=None):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                              validation_split=validation_split, class_weight=class_weight, verbose=2)



class TweetSentimentSeq2SeqGRU(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, third_layer_units = 128, third_layer_dropout = 0.5,
              relu_dense_layer = 64, dense_layer_units = 2):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        # X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(embeddings)
        #X = LSTM(first_layer_units, return_sequences=False, name='LSTM_1')(embeddings)
        X = GRU(128, return_sequences=False, name='GRU_1')(embeddings)

        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_1")(X)

        # Reshape to make it 3D
        #X = Reshape((self.max_sentence_len, 1))(X)
        X = Reshape((128, 1))(X)

        # Second LSTM Layer
        #X  = LSTM(second_layer_units, return_sequences=True, name="LSTM_2")(X)
        X  = GRU(64, return_sequences=True, name="GRU_2")(X)

        # Second Layer Dropout
        #X = LSTM(256, return_sequences=False, name="LSTM_2")(X)
        #X = Dropout(second_layer_dropout, name="DROPOUT_2")(X)
        # X = Dense(128, name="DENSE_1", activation='relu')(X)
        # Send to a Dense Layer with sigmoid activation
        #X = TimeDistributed(Dense(dense_layer_units, activation='relu', name="DENSE_2"))(X)
        X = Flatten()(X)
        X = Dense(128, activation='relu', name="DENSE_1")(X)

        # X = Activation("sigmoid", name="SIGMOID_1")(X)
        X = Dense(3, name="FINAL_DENSE")(X)
        X = Activation("softmax", name="SOFTMAX_1")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)

    def fit(self, X, Y, epochs=50, batch_size=32, shuffle=True, callbacks=None, validation_split=0.0,
            class_weight=None):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                              validation_split=validation_split, class_weight=class_weight, verbose=2)


class TweetSentimentSeq2SeqAttention(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, third_layer_units = 128, third_layer_dropout = 0.5,
              relu_dense_layer = 64, dense_layer_units = 2):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)

        # Attention
        attention = Permute((2, 1), name="Attention_Permute")(embeddings)
        attention = Dense(self.max_sentence_len, activation='softmax', name="Attention_Dense")(attention)
        attention_probs = Permute((2, 1), name='attention_probs')(attention)
        output_attention_mul = Multiply(name='attention_multiplu')([embeddings, attention_probs])

        # First LSTM Layer
        # X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(embeddings)
        #X = LSTM(first_layer_units, return_sequences=False, name='LSTM_1')(embeddings)
        X = LSTM(128, return_sequences=False, name='LSTM_1')(output_attention_mul)

        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_1")(X)

        # Reshape to make it 3D
        #X = Reshape((self.max_sentence_len, 1))(X)
        X = Reshape((128, 1))(X)

        # Second LSTM Layer
        #X  = LSTM(second_layer_units, return_sequences=True, name="LSTM_2")(X)
        X  = LSTM(128, return_sequences=True, name="LSTM_2")(X)

        # Second Layer Dropout
        #X = LSTM(256, return_sequences=False, name="LSTM_2")(X)
        #X = Dropout(second_layer_dropout, name="DROPOUT_2")(X)
        # X = Dense(128, name="DENSE_1", activation='relu')(X)
        # Send to a Dense Layer with sigmoid activation
        #X = TimeDistributed(Dense(dense_layer_units, activation='relu', name="DENSE_2"))(X)
        X = Flatten()(X)
        X = Dense(128, activation='elu', name="DENSE_1")(X)
        X = Dense(64, activation='elu', name="DENSE_2")(X)

        # X = Activation("sigmoid", name="SIGMOID_1")(X)
        X = Dense(3, name="FINAL_DENSE")(X)
        X = Activation("softmax", name="SOFTMAX_1")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)

    def fit(self, X, Y, epochs=50, batch_size=32, shuffle=True, callbacks=None, validation_split=0.0,
            class_weight=None):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                              validation_split=validation_split, class_weight=class_weight, verbose=2)
