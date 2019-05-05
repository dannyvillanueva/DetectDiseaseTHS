import numpy as np
from scipy import stats

import sys
import csv

def main(file_name=None):
    lens = []
    lens2 = []
    with open(file_name, "r", encoding="ISO-8859-1") as f:
        csv_file = csv.reader(f, delimiter=',')
        ones = 0
        zeros = 0
        twos = 0
        totals = 0
        for r in csv_file:
            lens.append(len(r[0]))
            lens2.append(len(r[0].split()))
            if r[1] == '0':
                zeros += 1
            elif r[1] == '1':
                ones += 1
            else:
                twos +=1
            totals +=1

        print("Zeros: ", zeros)
        print("Ones: ", ones)
        print("Twos: ", twos)
        verify_totals = zeros + ones + twos
        print("Totals: ", totals)
        print("Verified totals: ", verify_totals)
        print("Stats: ")
        print("% zeros: ", zeros/totals)
        print("% ones: ", ones/totals)
        print("% twos: ", twos/totals)
        print("Stats on tweet char len: ")
        print("Max len: ", np.max(lens))
        print("Avg len: ", np.average(lens))
        print("Median len: ", np.median(lens))
        print("Mode len: ", stats.mode(lens)[0])
        print("lens: ", lens[:10])
        print("Stats on tweet word len: ")
        print("Max len: ", np.max(lens2))
        print("Avg len: ", np.average(lens2))
        print("Median len: ", np.median(lens2))
        print("Mode len: ", stats.mode(lens2)[0])
        print("lens: ", lens2[:10])


if __name__ == "__main__":
    main("data/cleantextlabels3.csv")


