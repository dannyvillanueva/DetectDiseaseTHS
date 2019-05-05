import numpy as np
import csv
from ths.utils.contractions import expandContractions
import re
import string
import spacy
#from pattern.en import lemma as p_lemma
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ths.utils.similarity import Frobenius_Distance, TriUL_sim
from ths.utils.files import GloveEmbedding
from ths.utils.sentences import SentenceToIndices, SentenceToEmbeddingWithEPSILON
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import itertools
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer
import torch
from ths.test.models.models_fb import InferSent
import tensorflow as tf
import tensorflow_hub as hub
import time
import multiprocessing


def read_file(s):
    d = []
    y = []
    try:
        datafile = open("data/data" + s + "_labeled.csv", "r", encoding='cp1252')
        with datafile as f:
            for row in csv.reader(f):
                newline = " ".join(row[0].split())
                newline = newline.replace("’", "'")
                d.append(newline)
                y.append(int(row[1]))
    except Exception as e:
        print(e)
    #d_test = d[9000:12000]
    #y_test = y[9000:12000]
    return d, np.array(y)
    #return np.array(d_test), np.array(y_test)


def lemmatizer_spacy(d):
    # spacy.prefer_gpu()
    nlp = spacy.load('en_core_web_sm')
    a = []
    for row in d:
        row = " ".join([token.lemma_ for token in nlp(row)])
        a.append(row)
    return a


def lemmatizer_pattern(d):
    a = []
    for row in d:
        row = " ".join([p_lemma(wd) for wd in row.split()])
        a.append(row)
    return a


def lemmatizer(t, d):
    if t == 1:
        a = lemmatizer_spacy(d)
    if t == 2:
        a = lemmatizer_pattern(d)
    return a


def fix_text_format(d, l=None):
    a = []
    y = []
    count = 0
    for row in d:
        if len(row) >= 10 :
            text = row.lower().strip()  # remove white spaces
            text = expandContractions(text)  # expand contractions
            text = text.replace("“", '').replace("”", '').replace('"', '')  # remove quotation marks
            text = re.sub(r'http\S+', '', text)  # remove links
            text = re.sub(r'#\S+', '', text)  # remove hashtag
            text = re.sub(r'@\S+', '', text)  # remove mentions
            translator = str.maketrans('', '', string.punctuation)  # remove punctuation marks
            text = text.translate(translator)
            a.append(text)
            if l is not None:
                y.append(l[count])
        count += 1
    if l is None:
        return a
    return a, np.array(y)


def get_max_len(d):
    max_len = 0
    for line in d:
        a = line.strip()
        tweet_len = len(a)
        if tweet_len > max_len:
            max_len = tweet_len
    return max_len


def get_embedding_data(d, se):
    a = []
    max_len = get_max_len(d)
    for line in d:
        emb = se.map_sentence(line.lower(), max_len=max_len)
        a.append(emb)
    return np.array(a), max_len


def get_glove(glove_dims): # get glove embedding matrix
    if glove_dims == 50:
        G = GloveEmbedding("../test/data/glove.twitter.27B.50d.txt", dimensions=glove_dims)
    elif glove_dims == 100:
        G = GloveEmbedding("../test/data/glove.twitter.27B.100d.txt", dimensions=glove_dims)
    elif glove_dims == 300:
        G = GloveEmbedding("../test/data/glove.840B.300d.txt", dimensions=glove_dims)
    else:
        print("Wrong Number of dimensions")
        exit(0)
    word_to_idx, idx_to_word, embedding = G.read_embedding()
    SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
    return SE


def remove_stopwords(d):
    stop_words = set(stopwords.words('english'))
    a = []
    for i in d:
        word_tokens = word_tokenize(i)
        filtered_sentence = [w for w in word_tokens if w not in stop_words]
        newline = " ".join(filtered_sentence)
        a.append(newline)
    return a


def tokenizer_sentence_list(d):
    a = []
    for i in d:
        word_tokens = word_tokenize(i)
        a.append(word_tokens)
    return a


def worker(method, val1, val2, send_end):
    result = 0.0
    if method == 1:
        result = cosine_similarity(val1, val2)[0][0]
    if method == 2:
        result = Frobenius_Distance(val1, val2)
    if method == 3:
        result = TriUL_sim(val1, val2)
    send_end.send(result)


def jaccard_similarity_v1(d, y):
    start = time.time()
    t = 1  # type = 1: Spacy Lemmatizer, 2: Pattern Lemmatizer
    a = lemmatizer(t, d)
    print("data lemmatizer finished")
    i = 0
    final_map = []
    for r1 in a:
        r1 = set(r1.split())
        temp_map = []
        j = 0
        for r2 in a:
            if i != j:
                r2 = set(r2.split())
                i_len = len(r1.intersection(r2))
                u_len = len(r1.union(r2))
                jaccard_sim = round(i_len/u_len, 3)
                temp_map.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), jaccard_sim])
            j += 1
        temp_map = np.array(temp_map)
        temp_map = temp_map[temp_map[:, 4].argsort()[::-1]]
        temp_map = temp_map[:5]
        final_map.append(temp_map)
        i += 1
        if i%100 == 0:
            print("Row: ", i, "\n")
    end = time.time()
    print("jaccard similarity computed in ", (end - start)/60, " minutes")
    return final_map


def jaccard_similarity_v2(d, y):
    t = 1  # type = 1: Spacy Lemmatizer, 2: Pattern Lemmatizer
    a = lemmatizer(t, d)
    print("data lemmatizer finished")
    i = 0
    record = []
    final_map = []
    for r1 in a:
        r1 = set(r1.split())
        temp_map = []
        j = 0
        for r2 in a:
            if i > j:
                record_row = np.array(record[j])
                record_row = record_row[record_row[:, 2] == i]
                record_row =  record_row[0]
                temp_map.append([record_row[2], record_row[1], record_row[0], record_row[3], record_row[4]])
            if i < j:
                r2 = set(r2.split())
                i_len = len(r1.intersection(r2))
                u_len = len(r1.union(r2))
                jaccard_sim = round(i_len/u_len, 3)
                temp_map.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), jaccard_sim])
            j += 1
        record.append(temp_map)
        temp_map = np.array(temp_map)
        temp_map = temp_map[temp_map[:, 4].argsort()[::-1]]
        temp_map = temp_map[:5]
        final_map.append(temp_map)
        i += 1
        if i%100 == 0:
            print("Row: ", i, "\n")
    print("jaccard similarity computed")
    return final_map


def tfidf_similarity(d, y):
    start = time.time()
    tfidf_vectorizer = TfidfVectorizer()
    i = 0
    final_map_s1 = []
    final_map_s2 = []
    final_map_s3 = []
    for r1 in d:
        temp_map_s1 = []  # cosine similarity
        temp_map_s2 = []  # frobenius similarity
        temp_map_s3 = []  # triUL similarity
        j = 0
        for r2 in d:
            if i != j:
                documents = (r1, r2)
                tfidf_matrix = tfidf_vectorizer.fit_transform(documents).toarray()
                row1 = np.array(tfidf_matrix[0]).reshape(1, -1)
                row2 = np.array(tfidf_matrix[1]).reshape(1, -1)
                """
                jobs = []
                pipe_list = []
                for i in range(3):
                    recv, send = multiprocessing.Pipe(False)
                    if i == 0:
                        p = multiprocessing.Process(target=worker, args=(0, row1, row2, send))
                    if i == 1:
                        p = multiprocessing.Process(target=worker, args=(1, row1, row2, send))
                    if i == 2:
                        p = multiprocessing.Process(target=worker, args=(2, row1, row2, send))
                    jobs.append(p)
                    pipe_list.append(recv)
                    p.start()
                result_list = [x.recv() for x in pipe_list]
                """
                result_list = [cosine_similarity(row1, row2)[0][0], Frobenius_Distance(row1, row2), TriUL_sim(row1, row2)]
                temp_map_s1.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[0]])
                temp_map_s2.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[1]])
                temp_map_s3.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[2]])
            j += 1
        temp_map_s1 = np.array(temp_map_s1)
        temp_map_s2 = np.array(temp_map_s2)
        temp_map_s3 = np.array(temp_map_s3)
        temp_map_s1 = temp_map_s1[temp_map_s1[:, 4].argsort()[::-1]]
        temp_map_s2 = temp_map_s2[temp_map_s2[:, 4].argsort()[::-1]]
        temp_map_s3 = temp_map_s3[temp_map_s3[:, 4].argsort()[::-1]]
        temp_map_s1 = temp_map_s1[:5]
        temp_map_s2 = temp_map_s2[:5]
        temp_map_s3 = temp_map_s3[:5]
        final_map_s1.append(temp_map_s1)
        final_map_s2.append(temp_map_s2)
        final_map_s3.append(temp_map_s3)
        print("i: ", i)
        i += 1
        if i%100 == 0:
            print("Row: ", i, "\n")
    end = time.time()
    print("tf-idf similarity computed", (end - start)/60, " minutes")
    return final_map_s1, final_map_s2, final_map_s3


def average_embedding(d, y):
    i = 0
    final_map_s1 = []
    final_map_s2 = []
    final_map_s3 = []
    start = time.time()
    for r1 in d:
        temp_map_s1 = []  # cosine similarity
        temp_map_s2 = []  # frobenius similarity
        temp_map_s3 = []  # triUL similarity
        av_r1 = np.average(r1, axis=0)
        av1 = av_r1.reshape(1, -1)
        j = 0
        for r2 in d:
            if i != j:
                av_r2 = np.average(r2, axis=0)
                av2 = av_r2.reshape(1, -1)
                result_list = [cosine_similarity(av1, av2)[0][0], Frobenius_Distance(av1, av2),TriUL_sim(av1, av2)]
                temp_map_s1.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[0]])
                temp_map_s2.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[1]])
                temp_map_s3.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[2]])
            j += 1
        temp_map_s1 = np.array(temp_map_s1)
        temp_map_s2 = np.array(temp_map_s2)
        temp_map_s3 = np.array(temp_map_s3)
        temp_map_s1 = temp_map_s1[temp_map_s1[:, 4].argsort()[::-1]]
        temp_map_s2 = temp_map_s2[temp_map_s2[:, 4].argsort()[::-1]]
        temp_map_s3 = temp_map_s3[temp_map_s3[:, 4].argsort()[::-1]]
        temp_map_s1 = temp_map_s1[:5]
        temp_map_s2 = temp_map_s2[:5]
        temp_map_s3 = temp_map_s3[:5]
        final_map_s1.append(temp_map_s1)
        final_map_s2.append(temp_map_s2)
        final_map_s3.append(temp_map_s3)
        print("i: ", i)
        i += 1
        if i % 100 == 0:
            print("Row: ", i, "\n")
    end = time.time()
    print("average embedding similarity computed", (end - start) / 60, " minutes")
    return final_map_s1, final_map_s2, final_map_s3


def smooth_inverse_frequency(d, y, glove, m_l):
    d = fix_text_format(d)  # without labels: data = fix_text_format(data)
    print("Data cleaned")
    t = 1  # type = 1: Spacy Lemmatizer, 2: Pattern Lemmatizer
    d = lemmatizer(t, d)
    print("data lemmatizer finished")
    d = remove_stopwords(d)
    print("stop words removed")
    tokenized_list = tokenizer_sentence_list(d)
    word_counter = Counter(itertools.chain(*tokenized_list))
    se = get_glove(glove)
    print("glove loaded")
    start = time.time()
    a = 1e-3
    pre_emb = []
    for sentence in tokenized_list:
        token_length = len(sentence)
        for w in sentence:
            a_value = a / (a + word_counter[w]/len(word_counter))  # smooth inverse frequency, SIF
        vs = np.multiply(a_value, se.map_sentence(w, m_l))
        vs = vs.sum(axis=0)  # vs += sif * word_vector
        vs = vs / float(token_length)  # weighted average
        pre_emb.append(vs)
    [_, _, u] = np.array(svds(pre_emb, k=1))
    new_emb = []
    for v_s in pre_emb:
        v_s = v_s - v_s.dot(u * u.transpose())
        new_emb.append(v_s)
    print("SIF computed")
    i = 0
    final_map_s1 = []
    final_map_s2 = []
    final_map_s3 = []
    for r1 in new_emb:
        temp_map_s1 = []  # cosine similarity
        temp_map_s2 = []  # frobenius similarity
        temp_map_s3 = []  # triUL similarity
        r1 = r1.reshape(1, -1)
        j = 0
        for r2 in new_emb:
            if i != j:
                r2 = r2.reshape(1, -1)
                result_list = [cosine_similarity(r1, r2)[0][0], Frobenius_Distance(r1, r2), TriUL_sim(r1, r2)]
                temp_map_s1.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[0]])
                temp_map_s2.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[1]])
                temp_map_s3.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[2]])
            j += 1
        temp_map_s1 = np.array(temp_map_s1)
        temp_map_s2 = np.array(temp_map_s2)
        temp_map_s3 = np.array(temp_map_s3)
        temp_map_s1 = temp_map_s1[temp_map_s1[:, 4].argsort()[::-1]]
        temp_map_s2 = temp_map_s2[temp_map_s2[:, 4].argsort()[::-1]]
        temp_map_s3 = temp_map_s3[temp_map_s3[:, 4].argsort()[::-1]]
        temp_map_s1 = temp_map_s1[:5]
        temp_map_s2 = temp_map_s2[:5]
        temp_map_s3 = temp_map_s3[:5]
        final_map_s1.append(temp_map_s1)
        final_map_s2.append(temp_map_s2)
        final_map_s3.append(temp_map_s3)
        print("i: ", i)
        i += 1
        if i % 100 == 0:
            print("Row: ", i, "\n")
    end = time.time()
    print("Smooth Inverse Frequency similarity computed", (end - start) / 60, " minutes")
    return final_map_s1, final_map_s2, final_map_s3


def latent_semantic_indexing(d, y):
    start = time.time()
    vec = CountVectorizer()
    x = vec.fit_transform(d)
    df = pd.DataFrame(x.toarray(), columns=vec.get_feature_names())
    matrix = df.values
    #[v, s, u] = svds(matrix.astype("float64"), k=2)
    #u = u.T
    [v, s, u] = np.linalg.svd(matrix, full_matrices=True)
    uk = u[0:2].T
    vk = v[:, 0:2]
    sk = np.diag(s[0:2])
    # vtk = vk.T # To plot 2D
    sk_inverse = np.power(sk.diagonal(), -1)
    sk_inverse = np.diag(sk_inverse)  # s_k(-1)
    final_map_s1 = []
    final_map_s2 = []
    final_map_s3 = []
    i = 0
    for row in matrix:
        row = row.reshape((1, row.shape[0]))
        row_coordinate = np.dot(np.dot(row, uk), sk_inverse) # q = q.T x u_k x s_k(-1)
        temp_map_s1 = []  # cosine similarity
        temp_map_s2 = []  # frobenius similarity
        temp_map_s3 = []  # triUL similarity
        j = 0
        for e in vk:
            if i != j:
                ev_val = e.reshape((1, 2))
                result_list = [cosine_similarity(row_coordinate, ev_val)[0][0], Frobenius_Distance(row_coordinate, ev_val), TriUL_sim(row_coordinate, ev_val)]
                temp_map_s1.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[0]])
                temp_map_s2.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[1]])
                temp_map_s3.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[2]])
            j += 1
        temp_map_s1 = np.array(temp_map_s1)
        temp_map_s2 = np.array(temp_map_s2)
        temp_map_s3 = np.array(temp_map_s3)
        temp_map_s1 = temp_map_s1[temp_map_s1[:, 4].argsort()[::-1]]
        temp_map_s2 = temp_map_s2[temp_map_s2[:, 4].argsort()[::-1]]
        temp_map_s3 = temp_map_s3[temp_map_s3[:, 4].argsort()[::-1]]
        temp_map_s1 = temp_map_s1[:5]
        temp_map_s2 = temp_map_s2[:5]
        temp_map_s3 = temp_map_s3[:5]
        final_map_s1.append(temp_map_s1)
        final_map_s2.append(temp_map_s2)
        final_map_s3.append(temp_map_s3)
        print("i: ", i)
        i += 1
        if i % 100 == 0:
            print("Row: ", i, "\n")
    end = time.time()
    print("Latent Semantic Indexing similarity computed", (end - start) / 60, " minutes")
    return final_map_s1, final_map_s2, final_map_s3


def encoder_infersent(d, y):
    start = time.time()
    model_version = 1
    MODEL_PATH = "../test/data/infersent%s.pickle" % model_version
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    use_cuda = True
    model = model.cuda() if use_cuda else model
    W2V_PATH = '../test/data/glove.840B.300d.txt' if model_version == 1 else '../test/data/crawl-300d-2M.vec'
    model.set_w2v_path(W2V_PATH)
    model.build_vocab_k_words(K=100000)
    embeddings = model.encode(d, bsize=128, tokenize=False, verbose=True)
    print("InferSent embedding computed")
    i = 0
    final_map_s1 = []
    final_map_s2 = []
    final_map_s3 = []
    for r1 in embeddings:
        r1 = r1.reshape(1, -1)
        temp_map_s1 = []  # cosine similarity
        temp_map_s2 = []  # frobenius similarity
        temp_map_s3 = []  # triUL similarity
        j = 0
        for r2 in embeddings:
            r2 = r2.reshape(1, -1)
            if i != j:
                result_list = [cosine_similarity(r1, r2)[0][0],Frobenius_Distance(r1, r2), TriUL_sim(r1, r2)]
                temp_map_s1.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[0]])
                temp_map_s2.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[1]])
                temp_map_s3.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[2]])
            j += 1
        temp_map_s1 = np.array(temp_map_s1)
        temp_map_s2 = np.array(temp_map_s2)
        temp_map_s3 = np.array(temp_map_s3)
        temp_map_s1 = temp_map_s1[temp_map_s1[:, 4].argsort()[::-1]]
        temp_map_s2 = temp_map_s2[temp_map_s2[:, 4].argsort()[::-1]]
        temp_map_s3 = temp_map_s3[temp_map_s3[:, 4].argsort()[::-1]]
        temp_map_s1 = temp_map_s1[:5]
        temp_map_s2 = temp_map_s2[:5]
        temp_map_s3 = temp_map_s3[:5]
        final_map_s1.append(temp_map_s1)
        final_map_s2.append(temp_map_s2)
        final_map_s3.append(temp_map_s3)
        print("i: ", i)
        i += 1
        if i % 100 == 0:
            print("Row: ", i, "\n")
    end = time.time()
    print("InferSent similarity computed", (end - start) / 60, " minutes")
    return final_map_s1, final_map_s2, final_map_s3


def universal_sentence_encoder(d, y):
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    start = time.time()
    final_map_s1 = []
    final_map_s2 = []
    final_map_s3 = []
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings = sess.run(embed(d))
        print("universal encoder ready")
        i = 0
        for r1 in embeddings:
            r1 = r1.reshape(1, -1)
            temp_map_s1 = []  # cosine similarity
            temp_map_s2 = []  # frobenius similarity
            temp_map_s3 = []  # triUL similarity
            j = 0
            for r2 in embeddings:
                r2 = r2.reshape(1, -1)
                if i != j:
                    result_list = [cosine_similarity(r1, r2)[0][0], Frobenius_Distance(r1, r2), TriUL_sim(r1, r2)]
                    temp_map_s1.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[0]])
                    temp_map_s2.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[1]])
                    temp_map_s3.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[2]])
                j += 1
            temp_map_s1 = np.array(temp_map_s1)
            temp_map_s2 = np.array(temp_map_s2)
            temp_map_s3 = np.array(temp_map_s3)
            temp_map_s1 = temp_map_s1[temp_map_s1[:, 4].argsort()[::-1]]
            temp_map_s2 = temp_map_s2[temp_map_s2[:, 4].argsort()[::-1]]
            temp_map_s3 = temp_map_s3[temp_map_s3[:, 4].argsort()[::-1]]
            temp_map_s1 = temp_map_s1[:5]
            temp_map_s2 = temp_map_s2[:5]
            temp_map_s3 = temp_map_s3[:5]
            final_map_s1.append(temp_map_s1)
            final_map_s2.append(temp_map_s2)
            final_map_s3.append(temp_map_s3)
            print("i: ", i)
            i += 1
            if i % 100 == 0:
                print("Row: ", i, "\n")
        end = time.time()
    print("Universal Encoder similarity computed", (end - start) / 60, " minutes")
    return final_map_s1, final_map_s2, final_map_s3


def dense_ths_encoder(d, y, n_features, m_l):
    #load dense embedding
    max_len = m_l
    model_name = 'dense_t1'  # dense_t1, dense_t2, full_lstm, full_gru, gru_lstm
    auto_encoder, encoder = autoencoder(model_name, n_features, max_len)  # , init=kernel_init)
    new_emb = d
    i = 0
    final_map_s1 = []
    final_map_s2 = []
    final_map_s3 = []
    start = time.time()
    for r1 in new_emb:
        temp_map_s1 = []  # cosine similarity
        temp_map_s2 = []  # frobenius similarity
        temp_map_s3 = []  # triUL similarity
        av_r1 = np.average(r1, axis=0)
        av1 = av_r1.reshape(1, -1)
        j = 0
        for r2 in new_emb:
            if i != j:
                av_r2 = np.average(r2, axis=0)
                av2 = av_r2.reshape(1, -1)
                result_list = [cosine_similarity(av1, av2)[0][0], Frobenius_Distance(av1, av2), TriUL_sim(av1, av2)]
                temp_map_s1.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[0]])
                temp_map_s2.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[1]])
                temp_map_s3.append([round(i, 1), round(y[i], 1), round(j, 1), round(y[j], 1), result_list[2]])
            j += 1
        temp_map_s1 = np.array(temp_map_s1)
        temp_map_s2 = np.array(temp_map_s2)
        temp_map_s3 = np.array(temp_map_s3)
        temp_map_s1 = temp_map_s1[temp_map_s1[:, 4].argsort()[::-1]]
        temp_map_s2 = temp_map_s2[temp_map_s2[:, 4].argsort()[::-1]]
        temp_map_s3 = temp_map_s3[temp_map_s3[:, 4].argsort()[::-1]]
        temp_map_s1 = temp_map_s1[:5]
        temp_map_s2 = temp_map_s2[:5]
        temp_map_s3 = temp_map_s3[:5]
        final_map_s1.append(temp_map_s1)
        final_map_s2.append(temp_map_s2)
        final_map_s3.append(temp_map_s3)
        print("i: ", i)
        i += 1
        if i % 100 == 0:
            print("Row: ", i, "\n")
    end = time.time()
    print("average embedding similarity computed", (end - start) / 60, " minutes")
    return final_map_s1, final_map_s2, final_map_s3

if __name__ == "__main__":
    """
    Using labeled data 15k
    Calculate similarity with:
    1. Jaccard Similarity
    2. TF - IDF Similarity
    3. Average of Embeddings
    4. Smooth Inverse Frequency
    5. Latent Semantic Indexing
    6. Encoder InferSent (Facebook Encoder)
    7. Universal Sentence Encoder (Google Encoder)
    8. Dense THS Encoder 
    """
    # load data
    size = "15k"
    algorithm = 10
    # load embedding or load raw data
    data, labels = read_file(size)
    print("File read")
    glove_dims = 50  # options: 50, 200, 300
    load_embedding = 0
    if load_embedding:
        emb = np.load('../test/data/embedded_15k.npy')
        labels = np.load('../test/data/embedded_15k_labels.npy')
        print("data and labels loaded")
        max_len = len(emb[0])
    else:
        data, labels = fix_text_format(data, labels)  # without labels: data = fix_text_format(data)
        data_cleaned = data
        print("Data cleaned")
        data = lemmatizer(1, data)
        print("data lemmatizer finished")
        data = remove_stopwords(data)
        print("stop words removed")
        SE = get_glove(glove_dims)
        print("glove loaded")
        emb, max_len = get_embedding_data(data, SE)
        print("glove embedding set")
        np.save('../test/data/embedded_15k.npy', emb)
        np.save('../test/data/embedded_15k_labels.npy', labels)
        print("Embedded data saved")

    if algorithm == 1:
        jaccard = jaccard_similarity_v1(data, labels)
        outfile = '../test/output/output_jaccard_similarity.npy'
        np.save(outfile, jaccard)

    if algorithm == 2:
        s1, s2, s3 = tfidf_similarity(data, labels)
        outfile_s1 = '../test/output/output_tfidf_cosine_similarity.npy'
        outfile_s2 = '../test/output/output_tfidf_frobenius.npy'
        outfile_s3 = '../test/output/output_tfidf_triUl.npy'
        np.save(outfile_s1, s1)
        np.save(outfile_s2, s2)
        np.save(outfile_s3, s3)

    if algorithm == 3:
        s1, s2, s3 = average_embedding(emb, labels)
        outfile_s1 = '../test/output/output_average-embedding_cosine_similarity.npy'
        outfile_s2 = '../test/output/output_average-embedding_frobenius.npy'
        outfile_s3 = '../test/output/output_average-embedding_triUl.npy'
        np.save(outfile_s1, s1)
        np.save(outfile_s2, s2)
        np.save(outfile_s3, s3)

    if algorithm == 4:
        s1, s2, s3 = smooth_inverse_frequency(data, labels, glove_dims, max_len)
        outfile_s1 = '../test/output/output_smooth_inverse_frequency_cosine_similarity.npy'
        outfile_s2 = '../test/output/output_smooth_inverse_frequency_frobenius.npy'
        outfile_s3 = '../test/output/output_smooth_inverse_frequency_triUl.npy'
        np.save(outfile_s1, s1)
        np.save(outfile_s2, s2)
        np.save(outfile_s3, s3)

    if algorithm == 5:
        s1, s2, s3 = latent_semantic_indexing(data, labels)
        outfile_s1 = '../test/output/output_latent_semantic_indexing_cosine_similarity.npy'
        outfile_s2 = '../test/output/output_latent_semantic_indexing_frobenius.npy'
        outfile_s3 = '../test/output/output_latent_semantic_indexing_triUl.npy'
        np.save(outfile_s1, s1)
        np.save(outfile_s2, s2)
        np.save(outfile_s3, s3)

    if algorithm == 6:
        s1, s2, s3 = encoder_infersent(data, labels)
        outfile_s1 = '../test/output/output_encoder_infersent_cosine_similarity.npy'
        outfile_s2 = '../test/output/output_encoder_infersent_frobenius.npy'
        outfile_s3 = '../test/output/output_encoder_infersent_triUl.npy'
        np.save(outfile_s1, s1)
        np.save(outfile_s2, s2)
        np.save(outfile_s3, s3)

    if algorithm == 7:
        s1, s2, s3 = universal_sentence_encoder(data, labels)
        outfile_s1 = '../test/output/output_universal_sentence_encoder_cosine_similarity.npy'
        outfile_s2 = '../test/output/output_universal_sentence_encoder_frobenius.npy'
        outfile_s3 = '../test/output/output_universal_sentence_encoder_triUl.npy'
        np.save(outfile_s1, s1)
        np.save(outfile_s2, s2)
        np.save(outfile_s3, s3)

    if algorithm == 8:
        s1, s2, s3 = dense_ths_encoder(data, labels, glove_dims, max_len)
        outfile_s1 = '../test/output/output_dense_ths_encoder_cosine_similarity.npy'
        outfile_s2 = '../test/output/output_dense_ths_encoder_frobenius.npy'
        outfile_s3 = '../test/output/output_dense_ths_encoder_triUl.npy'
        np.save(outfile_s1, s1)
        np.save(outfile_s2, s2)
        np.save(outfile_s3, s3)

    print("Algorithm finished")