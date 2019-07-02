import json
from ths.utils.contractions import expandContractions
import re
import html
from autocorrect import spell
import random
import csv
import numpy as np
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.models import model_from_json
from keras.models import load_model

from ths.utils.files import GloveEmbedding
from ths.utils.sentences import SentenceToIndices, PadSentences
import tensorflow as tf
from keras.utils import to_categorical

import sys
#import os
#os.chdir(os.getcwd())

class TrimSentences:
    def __init__(self, trim_size):
        self.trim_size = trim_size

    def trim(self, sentence):
        if len(sentence) > self.trim_size:
            return sentence[:self.trim_size]
        else:
            return sentence

    def trim_list(self, sentence_list):
        result = []
        for s in sentence_list:
            temp = self.trim(s)
            result.append(temp)
        return result

def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

def fix_text_format(row):
    text = row.lower()  # lowercase text
    text = text.replace("\n", ' ')  # remove line break
    text = re.sub(r'http\S+', '', text)  # remove links
    text = re.sub(r'#\S+', '', text)  # remove hashtag
    text = re.sub(r'@\S+', '', text)  # remove mentions
    text = " ".join(text.split())  # remove duplicate spaces
    text = html.unescape(text)
    text = text.replace("“", ' ').replace("”", ' ').replace('"', ' ')  # remove quotation marks
    text = text.replace("’", "'")
    text = " ".join(reduce_lengthening(word) for word in text.split())
    text = expandContractions(text)  # expand contractions
    text = " ".join("i am" if word == "im" else word for word in text.split())
    text = re.sub(r"[^A-Za-z0-9 ]+", '', text)
    text = " ".join(spell(word) for word in text.split())
    text = " ".join(text.split())  # remove duplicate spaces
    text = text.lower()
    return text


def get_random_tweet(tl, item, disease_item):
    rand_item = random.choice(tl)
    max_rec = 1000
    counter = 0
    if disease_item in rand_item[0].split() and item != str(rand_item[0]):
        return rand_item
    else:
        counter +=1
        if counter < max_rec:
            get_random_tweet(tl, item, disease_item)
        else:
            return False


def get_disease(t):
    list_disease = ['zika','flu','ebola','measles','diarrhea']
    for d in list_disease:
        if d in t.split():
            return d
    return False


def has_disease(t):
    list_disease = ['zika','flu','ebola','measles','diarrhea']
    for d in list_disease:
        if d in t.split():
            return True
    return False


def lemmatizer_spacy(d):
    spacy.prefer_gpu()
    nlp = spacy.load('en_core_web_sm')
    row = " ".join([token.lemma_ for token in nlp(d)])
    #row = row.replace("-PRON-", '')  # remove -PRON-
    return row



def remove_stopwords(i):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(i)
    filtered_sentence = [w for w in word_tokens if w not in stop_words]
    newline = " ".join(filtered_sentence)
    return newline

def main1():
    with open("data/raw_tweet100k.json") as json_data:
        file = json_data.readlines()
        selected_data = file[20000:80000]
        cc = 0
        #selected_data = file[20000:21000]
        hash_tweet = {}
        for row in selected_data:
            full_tweet = json.loads(row)
            tweet_text = full_tweet["extended_tweet"]["full_text"]
            tweet_id = full_tweet['id_str']
            text_formatted = fix_text_format(tweet_text)
            if has_disease(text_formatted):
                hash_tweet[text_formatted] = tweet_id
                cc += 1
                print("cc: ", cc)
    tweet_list = []
    for k, v in hash_tweet.items():
        tweet_list.append([k, v])
    #tweet_list_final = tweet_list[:14999]
    tweet_list_final = tweet_list
    print(len(tweet_list_final), " sentences selected from ", len(hash_tweet))
    pair_sentences = []
    goal_tweets = 15000
    count = 0
    for i in tweet_list_final:
        if count < goal_tweets:
            disease = get_disease(str(i[0]))
            rand_value = get_random_tweet(tweet_list, str(i[0]), disease)
            if rand_value:
                print("pair(disease): ", disease, get_disease(rand_value[0]))
                pair_sentences.append([str(i[1]),str(i[0]), str(rand_value[1]), str(rand_value[0])])
                count +=1
            else:
                print("function do not found a pair tweet with the same disease")
    print("pair sentences array built: ", len(pair_sentences))
    slice = []
    labelers = 53
    counter = 1
    repeat_number = 3
    for row in pair_sentences:
        for _ in range(repeat_number):
            fields = [str(counter), str(row[0]), str(row[1]), str(row[2]), str(row[3])]
            counter += 1
            if counter > labelers:
                counter = 1
            slice.append(fields)
    print("sentences distribution built")
    #np.save('data/data_to_labeling', np.array(slice))
    np.save('data/data_to_labeling_test', np.array(slice))
    #outfile = open('data/data_to_labeling.csv', 'w', newline='')
    outfile = open('data/data_to_labeling_test.csv', 'w', newline='')
    writer = csv.writer(outfile)
    writer.writerow(["Classifier", "id1", "premise", "id2", "hypothesis"])
    writer.writerows(slice)
    outfile.close()
    print("process completed")


def main2():
    try:
        cc = 0
        datafile = open("data/cleantextlabels7.csv", "r")
        with datafile as f:
            hash_tweet = {}
            for row in csv.reader(f) :
                newline = " ".join(row[0].split())
                tweet_text = newline.replace("’", "'")
                text_formatted = fix_text_format(tweet_text)
                label =  int(row[1])
                if has_disease(text_formatted) and label == 1:
                    hash_tweet[text_formatted] = text_formatted
                    cc +=1
                    print("cc: ", cc)
    except Exception as e:
        print(e)

    tweet_list = []
    for k, v in hash_tweet.items():
        tweet_list.append([k, v])
    #tweet_list_final = tweet_list[:14999]
    tweet_list_final = tweet_list
    print(len(tweet_list_final), " sentences selected from ", len(hash_tweet))
    pair_sentences = []
    goal_tweets = 15000
    count = 0
    for i in tweet_list_final:
        if count < goal_tweets:
            disease = get_disease(str(i[0]))
            rand_value = get_random_tweet(tweet_list, str(i[0]), disease)
            if rand_value:
                print("pair(disease): ", disease, get_disease(rand_value[0]))
                pair_sentences.append([str(i[0]), str(rand_value[0])])
                count +=1
            else:
                print("function do not found a pair tweet with the same disease")
    print("pair sentences array built: ", len(pair_sentences))
    slice = []
    labelers = 53
    counter = 1
    repeat_number = 3
    for row in pair_sentences:
        for _ in range(repeat_number):
            fields = [str(counter), str(row[0]), str(row[1])]
            counter += 1
            if counter > labelers:
                counter = 1
            slice.append(fields)
    print("sentences distribution built")
    #np.save('data/data_to_labeling', np.array(slice))
    np.save('data/data_to_labeling_test', np.array(slice))
    #outfile = open('data/data_to_labeling.csv', 'w', newline='')
    outfile = open('data/data_to_labeling_test.csv', 'w', newline='')
    writer = csv.writer(outfile)
    writer.writerow(["Classifier", "premise", "hypothesis"])
    writer.writerows(slice)
    outfile.close()
    print("process completed")


def main3():
    try:
        cc = 0
        c= 0
        datafile = open("data/cleantextlabels7.csv", "r")
        with datafile as f:
            flu = set()
            zika = set()
            ebola = set()
            measles = set()
            diarrhea = set()
            for row in csv.reader(f) :
                #if cc < 100:
                newline = " ".join(row[0].split())
                tweet_text = newline.replace("’", "'")
                text_formatted = fix_text_format(tweet_text)
                label = int(row[1])
                if label == 1 and has_disease(text_formatted):
                    disease = get_disease(text_formatted)
                    if disease == 'flu':
                        flu.add(text_formatted)
                    if disease == 'zika':
                        zika.add(text_formatted)
                    if disease == 'ebola':
                        ebola.add(text_formatted)
                    if disease == 'measles':
                        measles.add(text_formatted)
                    if disease == 'diarrhea':
                        diarrhea.add(text_formatted)
                    cc += 1
                    print(cc, "from", c, "disease:", disease,"label:", label)
                c += 1
    except Exception as e:
        print(e)
    print("flu_len: ", len(flu))
    print("zika_len: ", len(zika))
    print("ebola_len: ", len(ebola))
    print("measles_len: ", len(measles))
    print("diarrhea_len: ", len(diarrhea))

    list_disease = ['zika', 'flu', 'ebola', 'measles', 'diarrhea']
    triplet = []
    for d in list_disease:
        if d == 'zika':
            list_set = zika
        if d == 'flu':
            list_set = flu
        if d == 'ebola':
            list_set = ebola
        if d == 'measles':
            list_set = measles
        if d == 'diarrhea':
            list_set = diarrhea
        temp_triplet = []
        if len(list_set) > 2:
            count = 1
            list_set = list(list_set)
            random.shuffle(list_set)
            if len(list_set) >= 1000:
                num_samples = 1000
            else:
                num_samples = len(list_set)
            for i in range(num_samples - 2):
                for j in range(num_samples - 2 - i):
                    for k in range(num_samples - j - 2 - i):
                        temp_triplet.append([list_set[i], list_set[j + i + 1], list_set[k + 2 + j + i]])
                print(d,"- step:", count, "/", num_samples)
                count += 1
            random.shuffle(temp_triplet)

            temp_triplet = temp_triplet[:1000]
            triplet = triplet+temp_triplet
        print("total triplets: ", d, len(temp_triplet))
    print("triplets sentences built: ", len(triplet))
    slice = []
    labelers = 53
    counter = 1
    repeat_number = 3
    for row in triplet:
        for _ in range(repeat_number):
            fields = [str(counter), str(row[0]), str(row[1]), str(row[2])]
            counter += 1
            if counter > labelers:
                counter = 1
            slice.append(fields)
    print("sentences distribution built")
    np.save('data/data_to_labeling_original3', np.array(triplet))
    np.save('data/data_to_labeling_test3', np.array(slice))
    #outfile = open('data/data_to_labeling.csv', 'w', newline='')
    outfile = open('data/data_to_labeling_test3.csv', 'w', newline='')
    writer = csv.writer(outfile)
    writer.writerow(["Classifier", "premise", "h1", "h2"])
    writer.writerows(slice)
    outfile.close()
    print("process completed")


def main4():
    try:
        data = load_data('data/data_to_labeling_test3.npy')
        set1 = data[:, 1]
        set2 = data[:, 2]
        set3 = data[:, 3]
        allsets = np.concatenate((set1, set2, set3), axis=0)

        cc = 0
        c = 0
        datafile = open("data/cleantextlabels7.csv", "r")
        with datafile as f:
            flu = set()
            zika = set()
            ebola = set()
            measles = set()
            diarrhea = set()
            for row in csv.reader(f):
                # if cc < 100:
                newline = " ".join(row[0].split())
                tweet_text = newline.replace("’", "'")
                text_formatted = fix_text_format(tweet_text)
                label = int(row[1])
                if label == 1 and has_disease(text_formatted) and text_formatted not in allsets:
                    disease = get_disease(text_formatted)
                    if disease == 'flu':
                        flu.add(text_formatted)
                    if disease == 'zika':
                        zika.add(text_formatted)
                    if disease == 'ebola':
                        ebola.add(text_formatted)
                    if disease == 'measles':
                        measles.add(text_formatted)
                    if disease == 'diarrhea':
                        diarrhea.add(text_formatted)
                    cc += 1
                    print(cc, "from", c, "disease:", disease, "label:", label)

                if cc % 10 == 0:
                    print("flu_len: ", len(flu))
                    print("zika_len: ", len(zika))
                    print("ebola_len: ", len(ebola))
                    print("measles_len: ", len(measles))
                    print("diarrhea_len: ", len(diarrhea))

                if len(flu) > 80 and len(measles) > 80 and len(diarrhea) > 80:
                    print("sets full")
                    break

                c += 1
    except Exception as e:
        print(e)
    print("flu_len: ", len(flu))
    print("zika_len: ", len(zika))
    print("ebola_len: ", len(ebola))
    print("measles_len: ", len(measles))
    print("diarrhea_len: ", len(diarrhea))

    list_disease = ['zika', 'flu', 'ebola', 'measles', 'diarrhea']
    triplet = []
    for d in list_disease:
        if d == 'zika':
            list_set = zika
        if d == 'flu':
            list_set = flu
        if d == 'ebola':
            list_set = ebola
        if d == 'measles':
            list_set = measles
        if d == 'diarrhea':
            list_set = diarrhea
        temp_triplet = []
        if len(list_set) > 2:
            count = 1
            list_set = list(list_set)
            random.shuffle(list_set)
            if len(list_set) >= 80:
                num_samples = 80
            else:
                num_samples = len(list_set)
            for i in range(num_samples - 2):
                for j in range(num_samples - 2 - i):
                    for k in range(num_samples - j - 2 - i):
                        temp_triplet.append([list_set[i], list_set[j + i + 1], list_set[k + 2 + j + i]])
                print(d, "- step:", count, "/", num_samples)
                count += 1
            random.shuffle(temp_triplet)

            temp_triplet = temp_triplet[:75]
            triplet = triplet + temp_triplet
        print("total triplets: ", d, len(temp_triplet))
    print("triplets sentences built: ", len(triplet))
    slice = []
    labelers = 3
    counter = 1
    repeat_number = 3
    for row in triplet:
        for _ in range(repeat_number):
            fields = [str(counter), str(row[0]), str(row[1]), str(row[2])]
            counter += 1
            if counter > labelers:
                counter = 1
            slice.append(fields)
    print("sentences distribution built")
    np.save('data/data_to_labeling_original4', np.array(triplet))
    np.save('data/data_to_labeling_test4', np.array(slice))
    # outfile = open('data/data_to_labeling.csv', 'w', newline='')
    outfile = open('data/data_to_labeling_test4.csv', 'w', newline='')
    writer = csv.writer(outfile)
    writer.writerow(["Classifier", "premise", "h1", "h2"])
    writer.writerows(slice)
    outfile.close()
    print("process completed")


def main5():
    try:
        datafile = open("data/cleantextlabels7.csv", "r")
        new_file = []
        with datafile as f:
            for row in csv.reader(f):
                newline = " ".join(row[0].split())
                tweet_text = newline.replace("’", "'")
                text_formatted = fix_text_format(tweet_text)
                label = int(row[1])
                new_file.append([text_formatted, label])
        np.save("data/cleantextlabels7", new_file)
    except FileNotFoundError as e:
        print(e)


def main6():
    try:
        datafile = open("data/final_similarity_data_new.csv", "r")
        new_file = []
        cc = 1
        with datafile as f:
            for row in csv.reader(f):
                print("row: ", cc)
                r0 = " ".join(row[0].split())
                r0 = remove_stopwords(r0)
                r0 = lemmatizer_spacy(r0)
                r0 = remove_stopwords(r0)
                r1 = int(row[1])
                r2 = int(row[2])

                r3 = " ".join(row[3].split())
                r3 = remove_stopwords(r3)
                r3 = lemmatizer_spacy(r3)
                r3 = remove_stopwords(r3)
                r4 = int(row[4])
                r5 = int(row[5])

                r6 = " ".join(row[6].split())
                r6 = remove_stopwords(r6)
                r6 = lemmatizer_spacy(r6)
                r6 = remove_stopwords(r6)
                r7 = int(row[7])
                r8 = int(row[8])

                r9 = float(row[9])
                r10 = float(row[10])
                r11 = int(row[11])

                new_row = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11]
                new_file.append(new_row)
                cc += 1
        np.save("data/final_similarity_data_new_lm", new_file)
        np_new_file = np.array(new_file)
        outfile = open('data/final_similarity_data_new_lm.csv', 'w', newline='')
        writer = csv.writer(outfile)
        writer.writerows(np_new_file)
        outfile.close()
        print("file final saved")

    except FileNotFoundError as e:
        print(e)

def save_file(st, sl, l):
    outfile = open(l+st+'.csv', 'w', newline='')
    writer = csv.writer(outfile)
    writer.writerow(["Classifier", "premise", "h1", "h2"])
    writer.writerows(sl)
    outfile.close()
    print("file student", st, "saved")


def save_multi_file_csv(l, d):
    student = '1'
    slice = []
    for row in d:
        if student != row[0]:
            save_file(student, slice, l)
            student = row[0]
            slice = []
        fields = [student, str(row[1]), str(row[2]), str(row[3])]
        slice.append(fields)
    save_file(student, slice, l)


def load_data(f):
    d = np.load(f)
    d = d[d[:, 0].argsort()]
    return d


def main7():

    zero = 0
    one = 0
    two = 0
    dict_tweets = {}
    all = np.load("data/cleantextlabels7.npy")
    for row in all:
        disease = get_disease(str(row[0]))
        if disease != 'zika' and has_disease(str(row[0])):
            dict_tweets[row[0]] = row[1]

    for key, value in dict_tweets.items():
        if int(value) == 0:
            zero += 1
        if int(value) == 1:
            one += 1
        if int(value) == 2:
            two += 1
    return zero, one, two

def main8():
    all = np.load("data/final_similarity_data_new.npy")
    label = all[:, 11]
    zero = 0
    one = 0

    for row in label:
        if int(row) == 0:
            zero +=1
        if int(row) == 1:
            one += 1
    return zero, one


def main9():
    with open("data/raw_tweet20k.json") as json_data:
        file = json_data.readlines()
        selected_data = file[:]
        new_file = []
        c = 0
        for row in selected_data:
            print(c)
            full_tweet = json.loads(row)
            tweet_text = full_tweet["extended_tweet"]["full_text"]
            tweet_id = full_tweet['id_str']
            tweet_text = tweet_text.replace("’", "'")
            text_formatted = fix_text_format(tweet_text)
            if has_disease(text_formatted):
                new_file.append([tweet_id, text_formatted])
            c +=1
    np.save("data/raw_tweet20k", new_file)
    return new_file


def set_trained_data(data, NN):
    new_data = []
    for row in data:
        new_data.append(row[1])

    G = GloveEmbedding("data/glove.6B.50d.txt", dimensions=50)
    word_to_idx, idx_to_word, embedding = G.read_embedding()
    S = SentenceToIndices(word_to_idx)

    X_Predict_Idx, max_len = S.map_sentence_list(new_data)
    i =0
    for s in X_Predict_Idx:
    #    print(str(i)+ ": ", s)
        i = i + 1

    #if max_len % 2 != 0:
    #    max_len = max_len + 1

    max_len = 72

    print("Max Len", max_len)

    P = PadSentences(max_len)
    Trim = TrimSentences(max_len)


    X_Predict_Final = P.pad_list(X_Predict_Idx)
    X_Predict_Final = Trim.trim_list(X_Predict_Final)
    X_Predict_Final = np.array(X_Predict_Final)
    X_Prediction = NN.predict(X_Predict_Final)
    final = np.argmax(X_Prediction, axis=1)
    return new_data, final

def main10():
    # Model reconstruction from JSON file
    with open('pretrained/modelcnnincepw6.json', 'r') as f:
        model = model_from_json(f.read())
    # Load weights into the new model
    model.load_weights('pretrained/modelcnnincepw6.h5')
    #model.summary()
    #Load data
    data = np.load("data/raw_tweet20k.npy")
    new_, final_ = set_trained_data(data, model)
    #print("Len data: ", len(new_))
    #print("Len prediction: ", len(final_))
    return [new_, final_]

def map_to_idx(S, X_words):
    X_indices, max_len = S.map_sentence_list(X_words)
    if max_len % 2 != 0:
        max_len = max_len + 1
    return X_indices, max_len

def binarize_aux(diseases, labels):
    D = to_categorical(diseases)
    L = to_categorical(labels)
    return np.hstack((D, L))

def set_disease(d):
    data = []
    diseases = {'flu': 0, 'measles': 1, 'diarrhea': 2, 'ebola': 3, 'zika': 4}
    for x in d:
        if not has_disease(x):
            print("No disease: ", x)
        dis_1 = get_disease(x)
        data.append(diseases[dis_1])
    return data

def set_prediction(pretain):
    data = pretain[:, 0]
    #Load Model
    model2 = load_model('trained/model1_50d_stoplemma_10e_new_prod.h5', custom_objects={'tf': tf})
    # summarize model.
    model2.summary()
    # Load data
    G = GloveEmbedding("data/glove.twitter.27B.50d.txt", dimensions=50)
    word_to_idx, idx_to_word, embedding = G.read_embedding()
    S = SentenceToIndices(word_to_idx)
    premise = "same busy and just over the flu so feeling great"
    premise = "when ebola struck the doctors stepped up to the plate and the rest of us sat and watched them do their stuff to all engineers and environmentalists this is our time to step up and find answers to these consequences of our failure to coexist with nature"
    premise = remove_stopwords(premise)
    premise = lemmatizer_spacy(premise)
    x_premise = remove_stopwords(premise)
    x_premise = np.full((len(data)), x_premise)

    x_hypothesis = []

    for row in data:
        #row = row.replace("’", "'")
        #row = fix_text_format(row)
        row = remove_stopwords(row)
        #row = lemmatizer_spacy(row)
        #row = remove_stopwords(row)
        x_hypothesis.append(row)
    x_hypothesis = np.array(x_hypothesis)

    X_one_indices, max_len1 = map_to_idx(S, x_premise)
    X_two_indices, max_len2 = map_to_idx(S, x_hypothesis)
    print("len: ", max_len1, max_len2)
    #max_len = max(max_len1, max_len2)
    max_len = 44
    print("max_len_final: ", max_len)

    P = PadSentences(max_len)
    Trim = TrimSentences(max_len)

    X_one_train = P.pad_list(X_one_indices)
    X_two_train = P.pad_list(X_two_indices)
    #X_one_train = Trim.trim_list(X_one_indices)
    X_two_train = Trim.trim_list(X_two_train)

    X_one_train = np.array(X_one_train)
    X_two_train = np.array(X_two_train)

    X_one_aux_disease = set_disease(x_premise)
    X_two_aux_disease = set_disease(x_hypothesis)

    new_dis = []
    for _ in range(len(data)):
        new_dis.append([0, 0, 1, 0, 0, 1])

    X_one_aux_train = new_dis

    X_two_aux_label = pretain[:, 1]

    #X_one_aux_train = binarize_aux(s4, X_one_aux_label)
    X_two_aux_train = binarize_aux(X_two_aux_disease, X_two_aux_label)
    #new_two = []
    #for row in range(len(data)):
    #    new_two.append([row[0], row[1], row[2], row[3], row[4], row[5], row[6]])

    X_two_aux_train = X_two_aux_train.tolist()
    for row in X_two_aux_train:
        del row[6]

    X_one_aux_train = np.array(X_one_aux_train)
    X_two_aux_train = np.array(X_two_aux_train)
    print("one_aux: ", np.array(X_one_aux_train).shape)
    print(X_one_aux_train[:5])
    print("two_aux: ", np.array(X_two_aux_train).shape)
    print(X_two_aux_train[:5])
    model2.load_weights('trained/model1_50d_stoplemma_10e_prod.h5')
    #model2.compile(optimizer='rmsprop',loss={'R1': 'mean_squared_error'},metrics={'R1': 'mse'}, loss_weights={'R1': 0.25})
    #model2.compile(optimizer='rmsprop')
    #model2.load_weights('trained/model1_50d_stoplemma_10e_prod.h5')
    X_Prediction = model2.predict([X_one_train, X_two_train, X_one_aux_train, X_two_aux_train])


    return X_Prediction

if __name__ == "__main__":

    #new_file = main10()
    #n = np.array(new_file)
    #t = n.transpose()
    #np.save("trained/pretrained_final", t)

    t = np.load("trained/pretrained_final.npy")
    prediction= set_prediction(t)

    new_pred = []
    c = 0
    for i in prediction:
        new_pred.append([c, i[0]])
        c += 1
    new_pred = np.array(new_pred)
    #sorted = [new_pred[:, 1].argsort()[::-1]]
    mat_sort = new_pred[new_pred[:, 1].argsort()[::-1]]
    top20 = mat_sort[:20]
    data = np.load("data/raw_tweet20k.npy")
    for i in top20:
        print("row: ", data[int(i[0])][1], " similarity index: ", i[1])
    np.save("trained/top20_final_v2", np.array(top20))
    #zero, one = main8()
    #a, b, c = main7()
    #data = load_data('data/data_to_labeling_test4.npy')
    #save_multi_file_csv('data/new_labeling_data_student_', data)


    print("Hello")


