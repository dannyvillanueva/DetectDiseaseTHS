import numpy as np
import csv
import math
from sklearn.utils import class_weight

def main(labeled_tweets_filename):
    np.random.seed(11)
    # open the file with tweets
    X_all = []
    Y_all = []
    All = []

    with open(labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
        i = 0
        csv_file = csv.reader(f, delimiter=',')
        ones_count = 0

        for r in csv_file:
            if i != 0:
                All.append(r)
            i = i + 1

    np.random.shuffle(All)

    ones_count = 0
    two_count = 0
    zero_count = 0
    for r in All:
        tweet = r[0]
        label = int(r[1])
        if (label == 0):
            zero_count +=1
        elif (label == 1):
            ones_count +=1
        else:
            two_count +=1
        # if (label == 2):
        #     label = 0
        # if (label == 1) and (ones_count <= 4611):
        #     X_all.append(tweet)
        #     Y_all.append(label)
        #     ones_count +=1
        # elif (label == 0):
        X_all.append(tweet)
        Y_all.append(label)

    print("len(Y_all): ", len(Y_all))
    class_weight_val = class_weight.compute_class_weight('balanced', np.unique(Y_all), Y_all)
    print("classes: ", np.unique(Y_all))
    print("counts for 0, 1, 2: ", zero_count, ones_count, two_count)
    print("class weight_val: ", class_weight_val)
    dictionary = { 0 : class_weight_val[0], 1 : class_weight_val[1], 2: class_weight_val[2]}
    print("dict: ", dictionary)

if __name__ == "__main__":
    main("data/cleantextlabels4.csv")