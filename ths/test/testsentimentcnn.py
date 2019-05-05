from ths.nn.sequences.processcnn import ProcessTweetsWord2VecOnePass2DCNNv2_1, ProcessTweetsWord2VecOnePass2DCNNv2_1Negate, \
    ProcessTweetsWord2VecOnePass2DCNN2Channelv2_1, ProcessTweetsWord2VecOnePass2DCNN2Channelv3, ProcessTweetsWord2VecOnePass2DCNN2Channelv5

def main():
    print("Working:")
    #P = ProcessTweetsWord2VecOnePass2DCNNv2_1("data/cleantextlabels3.csv", "trained/embedding3.csv")
    #P = ProcessTweetsWord2VecOnePass2DCNN2Channelv2_1("data/cleantextlabels4.csv", "data/glove.6B.50d.txt")
    P = ProcessTweetsWord2VecOnePass2DCNN2Channelv5("data/cleantextlabels5.csv", "data/glove.6B.50d.txt")

    #Bueno el model12cnnv2, ccn4, cnn5
    # inception 5 es bueno, 7 es bueno, 8 bueno, 9 es bueno modelinception9
    # inception con attention
    P.process("trained/modelinceptionatt1.json", "trained/modelinceptionatt1.h5", plot=True, epochs=30)
#joderme
if __name__ == "__main__":
    main()