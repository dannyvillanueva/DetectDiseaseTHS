import numpy as np
import csv

class ErrorAnalysis:
    @staticmethod
    def store_errors(X, Y, Y_Pred, file_name):
        errors = Y != Y_Pred
        errors = errors * 1 # trick to convert bool to 0s and 1s
        i = 0
        with open("data/" + file_name, "w") as f:
            out_f = csv.writer(f, delimiter=' ')
            for e in errors:
                if e:
                    data = []
                    tweet = X[i]
                    label = Y[i]
                    pred = Y_Pred[i]
                    data.append(tweet)
                    data.append(label)
                    data.append(pred)
                    out_f.writerow(data)

                i = i + 1
            f.flush()
