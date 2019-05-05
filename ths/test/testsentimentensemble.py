from ths.nn.sequences.processemsemble import ProcessTweetsWord2VecOnePassEnsemble

def main():
    print("Working:")
    #P = ProcessTweetsWord2VecOnePass2DCNNv2_1("data/cleantextlabels3.csv", "trained/embedding3.csv")
    P = ProcessTweetsWord2VecOnePassEnsemble("data/cleantextlabels3.csv", "data/glove.6B.50d.txt")

    #Bueno el model12cnnv2
    P.process("trained/modelensemble6.json", "trained/modelensemble6.h5", plot=True, epochs=20)
#joderme
if __name__ == "__main__":
    main()