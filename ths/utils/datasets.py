import csv
from collections import Counter

class TweetDataSetGenerator:
    def __init__(self, csv_file_name, vocabulary_size = 10000):
        self.csv_file_name = csv_file_name
        self.vocabulary_size = vocabulary_size

    def get_dataset(self):
        dictionary = {}
        reverse_dictionary = {}
        data = []

        with open(self.csv_file_name, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter = ',')
            for r in csv_file:
                if i !=0:
                    tweet = r[0]
                    #print(tweet)
                    data = data + tweet.split()
                i = i + 1

        # now build the dictionary
        # variable counts keeps track of the counts for the words

        counts  = [['UNK', 1]]
        # this variable counts now has the top most commond words found
        counts.extend(Counter(data).most_common(self.vocabulary_size -1))

        # now build the forward dictionary
        dictionary = {}
        for word, _ in counts:
            dictionary[word] = len(dictionary)

        dataset = []
        unk_count = 0
        for word in data:
            if word in dictionary:
                idx = dictionary[word]
            else:
                idx = 0
                unk_count += 1
            dataset.append(idx)

        #update number of times UNK was seen
        counts[0][1] = unk_count

        #build reverse dictionary
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        #return the indexed data along with the dictionaries
        return dataset, counts, dictionary, reverse_dictionary


