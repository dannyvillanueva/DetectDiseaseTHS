
import numpy as np
import sys

class EmbeddingException(Exception):
    pass

class GloveEmbedding:
    def __init__(self, filename, dimensions):
        if not filename :
            raise Exception("Illegal file name.")
        if dimensions < 1:
            raise Exception("Illegal value for dimensions")

        self.filename = filename
        self.dimensions = dimensions

    def get_dimensions(self):
        return self.dimensions

    def parse_embedding(self, data_vector):
        result = []
        for n in data_vector:
            result.append(float(n))
        return result

    def read_embedding_bad(self):
        try:
            data_in = open(self.filename, "r")
        except Exception as e:
            msg  = sys.exc_info()[0]
            raise EmbeddingException(msg) from e
        else:
            i = 0
            word_to_idx = {}
            idx_to_word = {}
            word_to_vect = []
            with data_in:
                for line in enumerate(data_in):
                    #print(line)
                    parts = line[1].split()
                    word_part = parts[0]
                    vector_parts = parts[1:]
                    idx_to_word[i] = word_part
                    word_to_idx[word_part] = i
                    i = i+ 1
                    word_to_vect.append(self.parse_embedding(vector_parts))
            #add <unk> token
            #unk = np.random.rand(self.dimensions,)
            unk = np.ones((self.dimensions,))
            idx_to_word[i] = "<unk>"
            word_to_idx["<unk>"] = i
            word_to_vect.append(unk)
            np_word_to_vect = np.array(word_to_vect)
            return word_to_idx, idx_to_word, np_word_to_vect

    def read_embedding(self):
        try:
            data_in = open(self.filename, "r",)
        except Exception as e:
            msg  = sys.exc_info()[0]
            raise EmbeddingException(msg) from e
        else:
            i = 0
            word_to_idx = {}
            idx_to_word = {}
            word_to_vect = []

            word_to_idx['<EOF>'] = 0
            idx_to_word[0] = None
            word_to_vect.append(np.asarray(np.zeros((self.dimensions,))))
            count_unk = 0
            with data_in:
                for line in enumerate(data_in):
                    #print(line)
                    parts = line[1].split()
                    word_part = parts[0]
                    vector_parts = parts[1:]
                    #vector_parts.shape((self.dimensions,))
                    vec = self.parse_embedding(vector_parts)
                    if len(vec) < self.dimensions:
                        error = self.dimensions - len(vec)
                        for _ in range(error):
                            vec.append(0.001)
                    if word_part == '<unk>' and count_unk == 0:
                        idx_to_word[i] = word_part
                        word_to_idx[word_part] = i
                        count_unk = 1
                        word_to_vect.append(np.asarray(vec))
                        i = i + 1
                    if word_part != '<unk>':
                        idx_to_word[i] = word_part
                        word_to_idx[word_part] = i
                        word_to_vect.append(np.asarray(vec))
                        i = i + 1
            np_word_to_vect = np.array(word_to_vect)
            #add <unk> token
            #unk = np.random.rand(self.dimensions,)
            #unk = np.ones((self.dimensions,))
            #idx_to_word[i] = "<unk>"
            #word_to_idx["<unk>"] = i
            #word_to_vect.append(unk)
            #np_word_to_vect = np.array(word_to_vect)
            return word_to_idx, idx_to_word, np_word_to_vect


class Word2VecEmbedding:
    def __init__(self, filename, dimensions):
        if not filename :
            raise Exception("Illegal file name.")
        if dimensions < 1:
            raise Exception("Illegal value for dimensions")

        self.filename = filename
        self.dimensions = dimensions

    def get_dimensions(self):
        return self.dimensions

    def parse_embedding(self, data_vector):

        result = []
        for n in data_vector:
            result.append(float(n))
        return result

    def read_embedding_bad(self):
        try:
            data_in = open(self.filename, "r")
        except Exception as e:
            msg  = sys.exc_info()[0]
            raise EmbeddingException(msg) from e
        else:
            i = 0
            word_to_idx = {}
            idx_to_word = {}
            word_to_vect = []
            with data_in:
                for line in enumerate(data_in):
                    #print(line)
                    parts = line[1].split()
                    word_part = parts[0]
                    vector_parts = parts[1:]
                    idx_to_word[i] = word_part
                    word_to_idx[word_part] = i
                    i = i+ 1
                    word_to_vect.append(self.parse_embedding(vector_parts))
            #add <unk> token
            unk = np.random.rand(self.dimensions,)
            idx_to_word[i] = "<unk>"
            word_to_idx["<unk>"] = i
            word_to_vect.append(unk)
            np_word_to_vect = np.array(word_to_vect)
            return word_to_idx, idx_to_word, np_word_to_vect

    def read_embedding(self):
        try:
            data_in = open(self.filename, "r")
        except Exception as e:
            msg  = sys.exc_info()[0]
            raise EmbeddingException(msg) from e
        else:
            i = 1
            word_to_idx = {}
            word_to_idx['<EOF>'] = 0
            idx_to_word = {}
            idx_to_word[0] = None
            word_to_vect = []
            word_to_vect.append(np.zeros((self.dimensions,)))
            with data_in:
                for line in enumerate(data_in):
                    #print(line)
                    parts = line[1].split()
                    word_part = parts[0]
                    vector_parts = parts[1:]
                    idx_to_word[i] = word_part
                    word_to_idx[word_part] = i
                    i = i+ 1
                    word_to_vect.append(self.parse_embedding(vector_parts))
            #add <unk> token
            unk = np.random.rand(self.dimensions,)
            idx_to_word[i] = "<unk>"
            word_to_idx["<unk>"] = i
            word_to_vect.append(unk)
            np_word_to_vect = np.array(word_to_vect)
            return word_to_idx, idx_to_word, np_word_to_vect




