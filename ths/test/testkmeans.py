from ths.utils.files import GloveEmbedding
from ths.utils.sentences import SentenceToIndices, SentenceToEmbeddingWithEPSILON, PadSentences
from ths.utils.similarity import matrix_cosine_similary, distance_similarity_matrix
import numpy as np;


def main():
    G = GloveEmbedding("../test/data/glove.6B.50d.txt")
    word_to_idx, idx_to_word, embedding = G.read_embedding()
    #print("locon: ", word_to_idx["locon"])
    print("Length dictionary: ", len(word_to_idx))
    #s = "I love New York and music locon"
    s = "The flu is making me sad"
    s = s.lower()
    print("Sentence: ", s)
    S = SentenceToIndices(word_to_idx)
    sentence = S.map_sentence(s)
    print("Sentence to indices: ", sentence)
    print("Padded: ", PadSentences(10).pad(sentence))
    SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
    matrix1 = SE.map_sentence(s, max_len=len(s))

    s2 = "The flu is making me sad".lower()
    matrix2 = SE.map_sentence(s2, max_len=len(s2))


    print("Matrix 1: ", matrix1)
    print("Matrix.shape: ", matrix1.shape)
    print("\n Matrix 2: ", matrix2)
    print("Matrix.shape: ", matrix2.shape)

    print("\n Self Similarity: ", matrix_cosine_similary(matrix1, matrix1))

    M1 = np.array([-1 , 40,0.04]).reshape((3,1))
    M2 = np.array([100 , 2 ,3 ]).reshape((3,1))
    print("M1: \n ", M1)
    print("M2: \n", M2)
    SimM = matrix_cosine_similary(M1, M2)
    print("SimM: \n", SimM)
    D = distance_similarity_matrix(SimM)
    print("D: ", D)

    M3 = np.array([[1, 2, 3,1], [4, 5, 6, 2], [7, 8, 9, 1]])
    M4 = np.array([[1, 2, 3.000001, 1], [4, 5, 6, 2], [7, 8, 9, 1]])

    SimM = matrix_cosine_similary(M3, M3)
    print("SimM: \n", SimM)
    D = distance_similarity_matrix(SimM)
    print("D: ", D)

    SimM = matrix_cosine_similary(M3, M4)
    print("\nSimM: \n", SimM)
    Up = np.triu(SimM)
    D = distance_similarity_matrix(SimM)
    print("D: ", D)
    print("Up: ", Up)
    print("sum Up: ", np.sum(Up))
    print("up I: ", np.triu(np.ones(Up.shape)))
    print("sum I: ", np.sum(np.triu(np.ones(Up.shape))))



    #print("Matrix Similarity: ", matrix_cosine_similary(matrix1, matrix2))
    #print("Similarity: ", distance_similarity_matrix(matrix_cosine_similary(matrix1, matrix2)))
    # print("Embedding i: ", embedding[word_to_idx["i"]])
    #
    # sentences = []
    # sentences.append("I esta malo".lower())
    # sentences.append("Love la musica salsa.".lower())
    # sentences.append("Uff, q mal te va nene".lower())
    # mapped, mlen = S.map_sentence_list(sentences)
    # print("mlen: ", mlen)
    # for s in mapped:
    #     print(s)
    #
    # print("Embedding 0: ", embedding[0])
    # print("Embedding 400001: ", embedding[400001])

if __name__ == "__main__":
    main()