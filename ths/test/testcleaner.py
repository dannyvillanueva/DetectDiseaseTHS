from ths.utils.cleaner import TweetCleaner
import csv

def main():
    input_name = "data/textlabels4.csv"
    output_name = "data/cleantextlabels4.csv"
    C = TweetCleaner(input_name=input_name, output_name=output_name)
    C.clean()
    print("Done")

    with open("data/cleantextlabels4.csv", "r", encoding="ISO-8859-1") as f:
        reader = csv.reader(f, delimiter = ',')
        max_len = 0
        max_sent = None
        length = 0.0
        counter = 0.0
        for row in reader:
            sentence = row[0].strip()
            length += len(sentence)
            counter +=1
            if len(sentence) > max_len:
                max_len = len(sentence)
                max_sent = sentence

        print("max_len: ", max_len)
        print("max_sent: ", max_sent)
        print("avg_length", length/counter)

if __name__ == "__main__":
    main()