from ths.nn.sequences.processlogistic import ProcessTweetsLogistic, ProcessTweetsConv1D
def main():
    print("Working:")
    P = ProcessTweetsConv1D("data/cleantextlabels5.csv", "trained/embedding3.csv")
    P.process(epochs=15)

if __name__ == "__main__":
        main()