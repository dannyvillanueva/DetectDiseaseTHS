import json
from ths.utils.contractions import expandContractions
import re
import html
from autocorrect import spell
import random
import csv
import numpy as np


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


def get_single_file(i):
    try:
        cc = 0
        datafile = open("data/labeling_data_student/labeling_data_student_"+str(i)+".csv", "r")
        relevance_input = []
        with datafile as f:
            for row in csv.reader(f):
                if cc > 0:
                    newline1 = " ".join(row[1].split())
                    newline2 = " ".join(row[2].split())
                    newline3 = " ".join(row[3].split())
                    if newline1.strip() == 'premise':
                        print("file ", i, "cc ", cc)
                    if row[4] != '' and row[5] != '' and newline3.strip() != "" and newline2.strip() != "" \
                            and newline1.strip() != "":
                        relevance_input.append([newline1, newline2, newline3, float(row[4]), float(row[5])])
                cc += 1
    except FileNotFoundError as e:
        print("No file ", i)
        return 0
    return relevance_input


def get_data():
    data = []
    for i in range(1,54):
        result = get_single_file(i)
        if result is not 0:
            data = data + result
    return np.array(data)


def average_relevance(data):
    new_set = set()
    new_data = []
    cc = 0
    for row in data:
        t0 = row[0]
        t1 = row[1]
        t2 = row[2]
        text_1 = t0+t1+t2
        temp_data = [row]
        if text_1 not in new_set:
            for i in range(cc+1, len(data)):
                row_2 = data[i]
                text_2 = row_2[0]+row_2[1]+row_2[2]
                if text_1 == text_2:
                    temp_data.append(row_2)
            #average
            temp_data = np.array(temp_data)
            r1 = temp_data[:, 3].astype(np.float).mean()
            r2 = temp_data[:, 4].astype(np.float).mean()
            new_triplet = [t0, t1, t2, r1, r2]
            new_set.add(text_1)
            new_data.append(new_triplet)
        cc += 1
    return new_data


def set_disease(d):
    data = []
    diseases = {'flu': 0, 'measles': 1, 'diarrhea': 2, 'ebola': 3, 'zika': 4}
    for x in d:
        dis_1 = get_disease(x[0])
        dis_2 = get_disease(x[1])
        dis_3 = get_disease(x[2])
        data.append([x[0], diseases[dis_1], x[1], diseases[dis_2], x[2], diseases[dis_3], x[3], x[4]])
    return data


def set_labels(d):
    #labeled_data = load_np_data('data/cleantextlabels7.npy')
    #new_data = dict(zip(labeled_data[:, 0], labeled_data[:, 1]))
    data = []

    for x in d:
        #key = x[0]
        #print(key)
        #label1 = new_data[key]
        #print(label1)
        #label2 = labeled_data_dict[x[2]]
        #label3 = labeled_data_dict[x[4]]
        data.append([x[0], x[1], 1, x[2], x[3], 1, x[4], x[5], 1, x[6], x[7]])
    return data


def set_global_relevance(d):
    data = []
    for x in d:
        x = list(x)
        if x[9]>= x[10]:
            r1 = 1
        else:
            r1= 0
        x.append(r1)
        data.append(x)
    return data


def load_np_data(f):
    d = np.load(f)
    d = d[d[:, 0].argsort()]
    return d


def get_sets_triplets(d):
    flu = []
    ebola = []
    measles = []
    diarrhea = []

    for x in d:
        disease = get_disease(x[0])
        if disease == 'flu':
            flu.append(x)
        if disease == 'ebola':
            ebola.append(x)
        if disease == 'measles':
            measles.append(x)
        if disease == 'diarrhea':
            diarrhea.append(x)
    return flu, ebola, measles, diarrhea


def get_sets_datalabels(dis, label=0):
    labeled_data = load_np_data('data/cleantextlabels7.npy')
    list_d = []
    for row in labeled_data:
        disease = get_disease(row[0])
        if disease == dis and int(row[1]) == 0:
            list_d.append(row)
    return list_d


def get_sets_datalabels_dif(dis):
    labeled_data = load_np_data('data/cleantextlabels7.npy')
    list_d = []
    for row in labeled_data:
        disease = get_disease(row[0])
        if disease and disease != dis and int(row[1]) == 0:
            list_d.append(row)
    return list_d


def set_input_same_class(d):
    flu, ebola, measles, diarrhea = get_sets_triplets(d)
    data = []

    list_disease = ['flu', 'ebola', 'measles', 'diarrhea']
    for dis in list_disease:
        for i in range(900):
            if dis == 'flu':
                triplet_all = random.choice(flu)
            if dis == 'ebola':
                triplet_all = random.choice(ebola)
            if dis == 'measles':
                triplet_all = random.choice(measles)
            if dis == 'diarrhea':
                triplet_all = random.choice(diarrhea)

            sets_datalabels = get_sets_datalabels(dis, 1)
            h3 = random.choice(sets_datalabels)
            list_dis = {'flu': 0, 'measles': 1, 'diarrhea': 2, 'ebola': 3, 'zika': 4}
            data.append([triplet_all[0],triplet_all[1], triplet_all[2], triplet_all[3], triplet_all[4],
                    triplet_all[5], h3[0], list_dis[dis], int(h3[1]), triplet_all[9], 0.05])
            i += 1
        print("disease", dis, "len data: ", len(data))
    data = set_global_relevance(data)
    new_data = d + data
    return new_data


def set_input_outof_class(d):
    flu, ebola, measles, diarrhea = get_sets_triplets(d)
    data = []
    triplet_set = set()
    list_disease = ['flu', 'ebola', 'measles', 'diarrhea']
    for dis in list_disease:
        for i in range(900):
            if dis == 'flu':
                triplet_all = random.choice(flu)
            if dis == 'ebola':
                triplet_all = random.choice(ebola)
            if dis == 'measles':
                triplet_all = random.choice(measles)
            if dis == 'diarrhea':
                triplet_all = random.choice(diarrhea)
            list_dis = {'flu': 0, 'measles': 1, 'diarrhea': 2, 'ebola': 3}
            sets_datalabels = get_sets_datalabels_dif(dis)
            h3 = random.choice(sets_datalabels)
            h3_dis = get_disease(h3[0])
            data.append([triplet_all[0],triplet_all[1], triplet_all[2], triplet_all[3], triplet_all[4],
                              triplet_all[5], h3[0], list_dis[h3_dis], int(h3[1]), triplet_all[9], 0.15])
            i += 1
        print("-out of class- disease", dis, "len data: ", len(data))
    data = set_global_relevance(data)
    new_data = d + data
    return new_data


def save_csv_final(final):
    np.save('data/final_similarity_data', final)
    outfile = open('data/final_similarity_data.csv', 'w', newline='')
    writer = csv.writer(outfile)
    writer.writerows(final)
    outfile.close()
    print("file final saved")


if __name__ == "__main__":
    final_data = get_data()
    print("finish data loading", len(final_data))
    final_data = average_relevance(final_data)
    print("finish average relevance", len(final_data))
    final_data = set_disease(final_data)
    print("finish set disease", len(final_data))
    final_data = set_labels(final_data)
    print("finish set labels", len(final_data))
    final_data = set_global_relevance(final_data)
    print("finish set global relevance", len(final_data))
    final_data = set_input_same_class(final_data)
    print("finish add triplets same class", len(final_data))
    final_data = set_input_outof_class(final_data)
    print("finish add triplets out of class", len(final_data))
    save_csv_final(final_data)