from ths.utils.datasets import TweetDataSetGenerator

def main():
    G = TweetDataSetGenerator("data/cleantextlabels2.csv")
    data, counts, dictionary, reverse_dictionary = G.get_dataset()

    print("data: ", data[:10])
    print("counts: ", counts[:10])
    print("dictionary: ", list(dictionary.keys())[:10], list(dictionary.values())[:10])
    print("rev dictionary: ", list(reverse_dictionary.keys())[:10], list(reverse_dictionary.values())[:10])
    print("len(dictionary)", len(dictionary))
if __name__ == "__main__":
    main()