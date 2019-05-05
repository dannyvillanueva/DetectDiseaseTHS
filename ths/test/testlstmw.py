from ths.nn.sequences.processlstmw import ProcessTweetsWord2VecOnePassLSTMv2_1, ProcessTweetsWord2VecTwoPassLSTMv2_1

def main():
    print("Working:")
    #P = ProcessTweetsWord2VecOnePass2DCNNv2_1("data/cleantextlabels3.csv", "trained/embedding3.csv")
    P = ProcessTweetsWord2VecTwoPassLSTMv2_1("data/cleantextlabels.csv", "data/glove.twitter.27B.50d.txt")
    #P = ProcessTweetsWord2VecTwoPassLSTMv2_1("data/cleantextlabels4.csv", "trained/embedding3-50d.csv")

    #Bueno el model12cnnv2
    # Excelente el de modellstmatt1 con attention
    # El mejor fue modellstmatt2 con attention
    # also good modellstmatt3
    # el 4 con dropout
    # 11 is good
    # 12 is good
    # 13 comes with attentions
    P.process("trained/modellstmatt15.json", "trained/modellstmatt15.h5", plot=False, epochs=140)
    # joder
#joderme
if __name__ == "__main__":
    main()