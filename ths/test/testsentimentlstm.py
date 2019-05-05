from ths.nn.sequences.processlstm import ProcessTweetsWord2VecOnePassLSTMv2_1

def main():
    print("Working:")
    #P = ProcessTweetsWord2VecOnePass2DCNNv2_1("data/cleantextlabels3.csv", "trained/embedding3.csv")
    #P = ProcessTweetsWord2VecOnePassLSTMv2_1("data/cleantextlabels5.csv", "data/glove.6B.50d.txt")
    P = ProcessTweetsWord2VecOnePassLSTMv2_1("data/cleantextlabels5.csv", "trained/embedding3-50d.csv")

    #Bueno el model12cnnv2
    # Excelente el de modellstmatt1 con attention
    # El mejor fue modellstmatt2 con attention
    # also good modellstmatt3
    # el 4 con dropout
    P.process("trained/modellstmatt7.json", "trained/modellstmatt7.h5", plot=True, epochs=15)
#joderme
if __name__ == "__main__":
    main()