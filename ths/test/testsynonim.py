from ths.utils.synomymos import OverSampleSynonym

def main():
    sentence1 = "I love a nice piece of chocolate."
    sentence2 = "Today is a sunny day."
    sentence3 = "Hell, yeah, me want that my man for tonight and next week."
    sentences = []
    print("sentence1.strip(): ", sentence1.strip())
    sentences.append(sentence1.strip().split())
    sentences.append(sentence2.strip().split())
    sentences.append(sentence3.strip().split())

    O = OverSampleSynonym()

    T = O.transform_sentences(sentences)
    for t in T:
        print(t)

    print("Done!")

if __name__ == "__main__":
    main()
