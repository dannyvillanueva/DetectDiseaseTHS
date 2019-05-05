from external.thesaurus import Word
from random import randint
import copy
import csv


class OverSampleSynonym:
    def __init__(self):
        pass

    def transform_sentence(self, sentence):
        # compute
        #print("sentence: ", sentence)
        sentence_len = len(sentence)
        idx = randint(0, sentence_len-1)
        word = sentence[idx]
        lookup = Word(word)
        synonyms = lookup.synonyms()
        syn_len = len(synonyms)
        #print("word: ", word, "synonyms: ", synonyms)
        if syn_len == 0:
            #No Synonym
            return -1, sentence
        idx2 = randint(0, syn_len-1)
        synonym = synonyms[idx2]
        result = copy.deepcopy(sentence)
        result[idx] = synonym
        return 0, result

    def transform_sentences(self, sentence_list, trials = 1):
        result = []
        for s in sentence_list:
            done = False
            new_sentence  = None
            while not done:
                code = 0
                code, new_sentence = self.transform_sentence(s)
                trials -= 1
                done = (code > 0) or (trials < 1)
            result.append(new_sentence)
        return result


class OverSampleTweetsZeroFile:
    def __init__(self, input_file_name, output_file_name):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name

    def fix(self, sentence):
        result = ""
        for s in sentence:
            result = result + s + " "
        result.strip()
        return result

    def oversample(self):
        Zeros = []
        with open(self.input_file_name, "r", encoding="ISO-8859-1") as data_in:
            with open(self.output_file_name, "w", encoding="ISO-8859-1") as data_out:
                csv_in = csv.reader(data_in, delimiter = ",")
                csv_out = csv.writer(data_out, delimiter = ",")
                ones_count = 0
                zeros_count = 0
                for r in csv_in:
                    tweet = r[0]
                    label = int(r[1])
                    if label == 0 or label == 2:
                        label = 0
                        zeros_count += 1
                        Zeros.append(r)
                    if label == 1:
                        ones_count += 1
                    new_r = []
                    new_r.append(tweet)
                    new_r.append(label)
                    csv_out.writerow(new_r)

                diff = ones_count - zeros_count
                zero_len = len(Zeros)
                over_sample = OverSampleSynonym()
                #print(ones_count, zeros_count, diff, zero_len)
                for _ in range(0, diff):
                    idx = randint(0, zero_len-1)
                    #print(idx)
                    temp = Zeros[idx]
                    old_sentence = temp[0].strip()
                    label = int(temp[1])
                    #print('old_sentence: ', old_sentence)
                    code, new_sentence = over_sample.transform_sentence(old_sentence.split())
                    new_sentence = self.fix(new_sentence)
                    new_r = []
                    new_r.append(new_sentence)
                    new_r.append(label)
                    csv_out.writerow(new_r)
        print("Done!")








