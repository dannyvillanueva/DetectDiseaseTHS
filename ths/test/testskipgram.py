from ths.nn.embedding.word2vec import SkipGrams
import numpy as np
from ths.utils.datasets import TweetDataSetGenerator

def main():
    #data = [1, 3, 4, 5, 4, 5, 10, 12, 100, 1]
    #data = np.random.randint(100, size = 1000000)
    T  = TweetDataSetGenerator("data/cleantextlabels2.csv")
    data, count, dictionary, reverse_dictionary = T.get_dataset()
    S = SkipGrams(text_data=data, dictionary_size=10000)
    targets, contexts, labels = S.build()
    print("targets: ", targets[:10])
    print("contexts: ", contexts[:10])
    print("labels: ", labels[:10])


if __name__ == "__main__":
    main()