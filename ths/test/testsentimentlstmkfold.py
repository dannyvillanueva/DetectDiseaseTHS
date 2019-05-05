from ths.nn.sequences.processexplstm import ProcessTweetsGloveLSTM2Layer

def main():
    print("Working:")
    P = ProcessTweetsGloveLSTM2Layer("data/cleantextlabels5.csv", "data/glove.6B.50d.txt")

    #Bueno el model12cnnv2
    P.process("trained/modellstm10.json", "trained/modellstm10.h5", plot=True, epochs=5)
#joderme
if __name__ == "__main__":
    main()