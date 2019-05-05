from keras.layers import Embedding, Input, Dense, Reshape, dot
from keras.models import Model
from keras.preprocessing.sequence import make_sampling_table, skipgrams

import numpy as np


class Word2VecNegSam:
    def __init__(self, vocabulary_size, vector_dim):
        self.vocabulary_size = vocabulary_size
        self.vector_dim = vector_dim
        self.model = None
        self.validation_model = None


    def build(self):
        # create the input layer for the target word and context word
        # shape is (1,) because each one is just an index of the
        # word in the vocabulary

        input_target = Input((1,), name="INPUT_TARGET")
        input_context = Input((1,), name="INPUT_CONTEXT")

        # now create the embedding layer
        # shape of this layer is vocabulary_size  x vector_dim x 1
        embedding = Embedding(input_dim=self.vocabulary_size, output_dim=self.vector_dim, input_length=1, name = "EMBEDDING")

        # now map input target and input context with the embedding
        # This will produce the mapping from index to embedding vector
        # this is what we want this model to lear, how to do this well
        target = embedding(input_target)
        context = embedding(input_context)

        # now to do a reshape to make sure the tensor have the
        # right dimension as a vector_dim x 1 tensor. This part I always
        # see is done, by barely explained why. In this case, it seems
        # is needed to make sure the dot product works correctly
        target = Reshape((self.vector_dim, 1))(target)
        context = Reshape((self.vector_dim, 1))(context)

        #compute the cosine similary between target and context
        # this is need by the validation model
        # axes 0 is used because similary will be done over 2 tensors
        similarity = dot([target, context], axes=0, normalize=True)

        # compute the dot product between target and context
        # axes is 1 becuase the axis = 0 is the batch size
        # the dot will be computed over batches of target,context pairs
        dot_product = dot([target, context], axes=1)
        dot_product = Reshape((1,))(dot_product)

        # now pass dot product throught sigmoid layer
        output_layer = Dense(1, activation='sigmoid', name='SIGMOID')(dot_product)

        #now build the model
        self.model = Model(input=[input_target, input_context], output=output_layer)

        self.validation_model = Model(input=[input_target, input_context], output=similarity)

        return self.model

    def compile(self, optimizer='rmsprop', metrics=['accuracy']):
        self.model.compile(loss= 'binary_crossentropy', optimizer=optimizer, metrics = metrics)

    def summary(self):
        self.model.summary()

    def get_model(self):
        return self.model

    def get_validation_model(self):
        return self.validation_model

    def train(self, X_targets, X_contexts, Y, epochs = 1000, callback = None):
        x_target = np.zeros((1,))
        x_context = np.zeros((1,))
        y_label = np.zeros((1,))
        for cnt in range(epochs):
            Y_size = len(Y)
            idx = np.random.randint(0, Y_size - 1)
            x_target[0,] = X_targets[idx]
            x_context[0,] = X_contexts[idx]
            y_label[0,] = Y[idx]
            loss = self.model.train_on_batch([x_target, x_context], y_label)
            if cnt % 1000 == 0:
                print("Iteration {}, loss={}".format(cnt, loss))
            if cnt %10000 == 0:
                if callback:
                    callback()

        return self.model.get_layer('EMBEDDING').get_weights()


class SkipGrams:
    def __init__(self, text_data, dictionary_size, window_size=3):
        self.dictionary_size = dictionary_size
        self.text_data = text_data
        self.window_size = window_size

    def build(self):
        sampling_table = make_sampling_table(self.dictionary_size)
        word_pairs, labels = skipgrams(self.text_data, self.dictionary_size, window_size=self.window_size,
                                       sampling_table=sampling_table)
        #print(word_pairs)
        word_targets, word_contexts = zip(*word_pairs)
        word_contexts = np.array(word_contexts, dtype="int32")
        word_targets = np.array(word_targets, dtype="int32")
        labels = np.array(labels, dtype="int32")
        return word_contexts, word_targets, labels


class Word2VecValidationCallback:
    def __init__(self, reverse_dictionary, validation_model, valid_size, valid_window, top_k):
        self.reverse_dictionary = reverse_dictionary
        self.vocabulary_size = len(self.reverse_dictionary)
        self.validation_model = validation_model
        self.valid_size = valid_size
        self.valid_window = valid_window
        self.valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        self.top_k = top_k

    def validate(self):
        for i in range(self.valid_size):
            valid_word = self.reverse_dictionary[self.valid_examples[i]]
            similary = self.get_similarity(self.valid_examples[i])
            nearest = (-similary).argsort()[1:self.top_k+1]
            log_msg = 'Nearest words to %s: ' % valid_word
            for k in range(self.top_k):
                close_word = self.reverse_dictionary[nearest[k]]
                log_msg = '%s %s, ' % (log_msg, close_word)
            print(log_msg)

    def get_similarity(self, valid_word_idx):
        sim = np.zeros((self.vocabulary_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in range(self.vocabulary_size):
            in_arr1[0,] = valid_word_idx
            in_arr2[0,] = i
            out = self.validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim

    def __call__(self, *args, **kwargs):
        self.validate()
