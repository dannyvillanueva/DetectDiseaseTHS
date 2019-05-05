import json
from ths.utils.contractions import expandContractions
import re
import html
from autocorrect import spell
import random
import csv
import numpy as np
import sys
#import os
#os.chdir(os.getcwd())

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
        cc = 0
        datafile = open("data/cleantextlabels7.csv", "r")
        with datafile as f:
            flu = set()
            zika = set()
            ebola = set()
            measles = set()
            diarrhea = set()
            for row in csv.reader(f):
                newline = " ".join(row[0].split())
                tweet_text = newline.replace("’", "'")
                text_formatted = fix_text_format(tweet_text)
                if has_disease(text_formatted) and int(row[1]) == 1:
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
                    cc +=1
                    print("cc: ", cc)
    except Exception as e:
        print(e)

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
        i = 0
        j = 0
        for i in list_set:
            for j in list_set:
                if (i != j):
                    triplet.append([i, j])
        print("triplets of disease: ", d, " finished")
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
    np.save('data/data_to_labeling_original4', np.array(triplet))
    np.save('data/data_to_labeling_test4', np.array(slice))
    #outfile = open('data/data_to_labeling.csv', 'w', newline='')
    outfile = open('data/data_to_labeling_test4.csv', 'w', newline='')
    writer = csv.writer(outfile)
    writer.writerow(["Classifier", "premise", "hypothesis"])
    writer.writerows(slice)
    outfile.close()
    print("process completed")


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

def load_data(f):
    d = np.load(f)
    d = d[d[:, 0].argsort()]
    return d


if __name__ == "__main__":
    #main3()
    data = load_data('data/data_to_labeling_test3.npy')
    save_multi_file_csv('data/labeling_data_student_', data)