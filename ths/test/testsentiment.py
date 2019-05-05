from ths.nn.sequences.process import ProcessTweetsGlove, ProcessTweetsGloveOnePass, ProcessTweetsGloveOnePassSM, ProcessTweetsWord2VecOnePassSM, ProcessTweetsWord2VecOnePassCNN, ProcessTweetsWord2VecOnePass2DCNN, ProcessTweetsWord2VecOnePass2DCNNv2, ProcessTweetsWord2VecOnePass2DCNNv4, ProcessTweetsWord2VecOnePassSMv2

def main():
    print("Working:")
    P  = ProcessTweetsGlove("data/cleantextlabels.csv","data/glove.6B.50d.txt")
    #P  = ProcessTweetsGloveOnePass("data/cleantextlabels.csv","data/glove.6B.50d.txt")
    #P = ProcessTweetsGloveOnePassSM("data/cleantextlabels3.csv", "data/glove.6B.50d.txt")
    #P = ProcessTweetsGloveOnePass("data/cleantextlabels3.csv", "data/glove.6B.50d.txt")

    #P = ProcessTweetsWord2VecOnePassSM("data/cleantextlabels3.csv", "trained/embedding3.csv")
    #P = ProcessTweetsWord2VecOnePass2DCNN("data/cleantextlabels3.csv", "trained/embedding3.csv")
    #P = ProcessTweetsWord2VecOnePassCNN("data/cleantextlabels3.csv", "trained/embedding3.csv")
    #P = ProcessTweetsWord2VecOnePass2DCNNv2("data/cleantextlabels3.csv", "trained/embedding3.csv")
    #P = ProcessTweetsWord2VecOnePass2DCNNv4("data/cleantextlabels3.csv", "trained/embedding3.csv")
    #P = ProcessTweetsWord2VecOnePassSMv2("data/cleantextlabels3.csv", "trained/embedding3.csv")

    #Bueno el model12cnnv2
    P.process("trained/model12lstm.json", "trained/model12lstm.h5")
#joderme
if __name__ == "__main__":
    main()