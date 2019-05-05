from ths.utils.files import GloveEmbedding
from ths.utils.sentences import SentenceToIndices, SentenceToEmbeddingWithEPSILON, SentenceToEmbedding, PadSentences
from ths.utils.similarity import matrix_cosine_similary, distance_similarity_matrix, Frobenius_Distance, TriUL_sim
import numpy as np
import random


def main():
    pass

def max_len_three(file1, file2, file3):
    try:
        data_one = open(file1, "r", encoding='utf-8')
        data_two = open(file2, 'r', encoding='utf-8')
        data_three = open(file3, 'r', encoding='utf-8')
    except Exception as e:
        print(e)
    else:
        max_len1 = 0
        max_len2 = 0
        max_len3 = 0
        SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
        with data_one:
            for line in data_one:
                matrix = SE.map_sentence(line.lower())
                tweet_len = len(matrix)
                if tweet_len > max_len1:
                    max_len1 = tweet_len
        with data_two:
            for line in data_two:
                matrix = SE.map_sentence(line.lower())
                tweet_len = len(matrix)
                if tweet_len > max_len2:
                    max_len2 = tweet_len
        with data_three:
            for line in data_three:
                matrix = SE.map_sentence(line.lower())
                tweet_len = len(matrix)
                if tweet_len > max_len3:
                    max_len3 = tweet_len
        return max(max_len1, max_len2, max_len3)

def get_max_len(file):
    max_len = 0
    SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
    for line in file:
        matrix = SE.map_sentence(line.lower())
        tweet_len = len(matrix)
        if tweet_len > max_len:
            max_len = tweet_len
    return max_len

def setCentroidsFromLabel(file1, file2, file3, max_len):
    cluster1 = []
    cluster2 = []
    cluster3 = []
    try:
        data_one = open(file1, "r", encoding='utf-8')
        data_two = open(file2, 'r', encoding='utf-8')
        data_three = open(file3, 'r', encoding='utf-8')
    except Exception as e:
        print(e)
    else:
        SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
        for line in data_one:
            matrix = SE.map_sentence((line.lower()), max_len=max_len)
            cluster1.append(matrix)
        with data_two:
            for line in data_two:
                matrix = SE.map_sentence((line.lower()), max_len=max_len)
                cluster2.append(matrix)
        with data_three:
            for line in data_three:
                matrix = SE.map_sentence((line.lower()), max_len=max_len)
                cluster3.append(matrix)
        #Set centroids
        centroid1 = (1 / len(cluster1)) * np.sum(cluster1, axis=0)
        centroid2 = (1 / len(cluster2)) * np.sum(cluster2, axis=0)
        centroid3 = (1 / len(cluster3)) * np.sum(cluster3, axis=0)
    return centroid1, centroid2, centroid3


def setCentroidsRandom(finaldata, max_len):
    i = random.choice(list(enumerate(finaldata)))[0]
    c1 = finaldata[i]
    j = random.choice(list(enumerate(finaldata)))[0]
    c2 = finaldata[j]
    k = random.choice(list(enumerate(finaldata)))[0]
    c3 = finaldata[k]
    print("centroid 1: ", data[i])
    print("centroid 2: ", data[j])
    print("centroid 3: ", data[k])
    if (i != j) and (i!= k) and (k != j):
        return c1, c2, c3
    else:
        setCentroidsRandom(data, max_len)


def clusterAsignment(finaldata, c1, c2, c3):
    cluster1 = []
    cluster2 = []
    cluster3 = []
    for line in finaldata:
        sim1 = matrix_cosine_similary(c1, line)
        sim2 = matrix_cosine_similary(c2, line)
        sim3 = matrix_cosine_similary(c3, line)
        S1 = distance_similarity_matrix(sim1)
        S2 = distance_similarity_matrix(sim2)
        S3 = distance_similarity_matrix(sim3)
        m = min(S1,S2,S3)
        if(m==S1):
            cluster1.append(line)
        elif(m==S2):
            cluster2.append(line)
        else:
            cluster3.append(line)
    return cluster1, cluster2, cluster3


def clusterAsignmentv2(finaldata, c1, c2, c3):
    cluster1 = []
    cluster2 = []
    cluster3 = []
    dictionary = {}
    i = 0
    for line in finaldata:
        sim1 = matrix_cosine_similary(c1, line)
        sim2 = matrix_cosine_similary(c2, line)
        sim3 = matrix_cosine_similary(c3, line)
        S1 = distance_similarity_matrix(sim1)
        S2 = distance_similarity_matrix(sim2)
        S3 = distance_similarity_matrix(sim3)
        m = min(S1,S2,S3)
        if(m==S1):
            cluster1.append(line)
            dictionary[data[i]] = 1
        elif(m==S2):
            cluster2.append(line)
            dictionary[data[i]] = 2
        else:
            cluster3.append(line)
            dictionary[data[i]] = 3
        i += 1
    return cluster1, cluster2, cluster3, dictionary


def clusterAsignmentv3(finaldata, c1, c2, c3):
    cluster1 = []
    cluster2 = []
    cluster3 = []
    dictionary = {}
    i = 0
    for line in finaldata:
        sim1 = TriUL_sim(c1, line)
        sim2 = TriUL_sim(c2, line)
        sim3 = TriUL_sim(c3, line)
        m = min(sim1, sim2, sim3)
        if(m==sim1):
            cluster1.append(line)
            dictionary[data[i]] = 1
        elif(m==sim2):
            cluster2.append(line)
            dictionary[data[i]] = 2
        else:
            cluster3.append(line)
            dictionary[data[i]] = 3
        i += 1
    return cluster1, cluster2, cluster3, dictionary

def clusterAsignmentv4(finaldata, c1, c2, c3):
    cluster1 = []
    cluster2 = []
    cluster3 = []
    dictionary = {}
    i = 0
    for line in finaldata:
        sim1 = Frobenius_Distance(c1, line)
        sim2 = Frobenius_Distance(c2, line)
        sim3 = Frobenius_Distance(c3, line)
        m = min(sim1, sim2, sim3)
        if(m==sim1):
            cluster1.append(line)
            dictionary[data[i]] = 1
        elif(m==sim2):
            cluster2.append(line)
            dictionary[data[i]] = 2
        else:
            cluster3.append(line)
            dictionary[data[i]] = 3
        i += 1
    return cluster1, cluster2, cluster3, dictionary

def moveCentroid(cluster1, cluster2, cluster3, c1, c2, c3):
    sumc1 = np.sum(cluster1, axis=0)
    sumc2 = np.sum(cluster2, axis=0)
    sumc3 = np.sum(cluster3, axis=0)
    if(len(cluster1)>0):
        newc1 = (1.0 / len(cluster1)) * sumc1
        c1 = newc1
    if(len(cluster2)>0):
        newc2 = (1.0 / len(cluster2)) * sumc2
        c2 = newc2
    if(len(cluster3)>0):
        newc3 = (1.0 / len(cluster3)) * sumc3
        c3 = newc3
    return c1, c2, c3

#c = centroid
def centroidsSimilarity(c1, oldc1, c2, oldc2, c3, oldc3):
    if (oldc1 == c1).all() and (oldc2 == c2).all() and (oldc3 == c3).all():
        return True
    else:
        return False


#cl = cluster
def clustersSimilarity(cl1, oldcl1, cl2, oldcl2, cl3, oldcl3):
    if np.array_equal(cl1, oldcl1) and np.array_equal(cl2, oldcl2) and np.array_equal(cl3, oldcl3):
        return True
    else:
        return False


def epsilonSimilarity(c1, oldc1, c2, oldc2, c3, oldc3):
    print("Entered in empsilonSimilarity")
    EPSILON_VALUE = 0.01

    shape = np.array(c1).shape
    #cluster 1
    ans1 = (TriUL_sim(c1, oldc1) < EPSILON_VALUE )
    print("Epsilon 1: ", ans1)

    #cluster 2
    ans2 = (TriUL_sim(c2, oldc2)< EPSILON_VALUE)
    print("Epsilon 2: ", ans2)

    #cluster 3
    ans3 = (TriUL_sim(c3, oldc3) < EPSILON_VALUE )
    print("Epsilon 3: ", ans3)

    if ans1 and ans2 and ans3:
        return True
    else:
        return False


if __name__ == "__main__":
    #Step 1: Set Centroids
    print("Step 1: Starting")
    G = GloveEmbedding("../test/data/glove.twitter.27B.50d.txt")
    word_to_idx, idx_to_word, embedding = G.read_embedding()
    S = SentenceToIndices(word_to_idx)
    SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
    data = []
    dictionary1  = {}
    dictionary2  = {}
    try:
        datafile = open("data/tweets.txt", "r", encoding='utf-8')
        with datafile as f:
            for line in f:
                newline = " ".join(line.split())
                data.append(newline)
    except Exception as e:
        print(e)
    max_len = get_max_len(data)
    finaldata = []
    for line in data:
        emb = SE.map_sentence(line.lower(), max_len=max_len)
        finaldata.append(emb)
        dictionary1[line] = emb

    #c1, c2, c3 = setCentroidsFromLabel("data/clusterone.txt", "data/clustertwo.txt", "data/clusterthree.txt", max_len)
    c1, c2, c3 = setCentroidsRandom(finaldata, max_len)
    print("Step 1: passed")
    #Step 2: Cluster Asignment
    n = 0
    oldc1 = c1
    oldc2 = c2
    oldc3 = c3
    oldcluster1 = []
    oldcluster2 = []
    oldcluster3 = []
    next = True

    type = 1 #1: TriUL_sim(), 2: Frobenius_Distance()
    while(next):
        print("Step 2: Starting")
        #cluster1, cluster2, cluster3 = clusterAsignment(finaldata, oldc1, oldc2, oldc3)
        if type == 1:
            cluster1, cluster2, cluster3, dictionary2 = clusterAsignmentv3(finaldata, oldc1, oldc2, oldc3)
        if type == 2:
            cluster1, cluster2, cluster3, dictionary2 = clusterAsignmentv3(finaldata, oldc1, oldc2, oldc3)
        #cluster1, cluster2, cluster3 = clusterAsignment(finaldata, oldc1, oldc2, oldc3)
        print("Step 2: passed")
        #Step 3: Move Clusters
        print("Step 3: Starting")
        c1, c2, c3 = moveCentroid(cluster1, cluster2, cluster3, oldc1, oldc2, oldc3)
        n+=1
        print("Step 3: passed")
        print("Simulation #:", n, "finished")
        #Step 4: Calculate similarity of clusters and centroids
        if n < 50:
            # result = centroidsSimilarity(c1, oldc1, c2, oldc2, c3, oldc3)
            result = clustersSimilarity(cluster1, oldcluster1, cluster2, oldcluster2, cluster3, oldcluster3)
        else:
            result = epsilonSimilarity(c1, oldc1, c2, oldc2, c3, oldc3)

        if(result):
            next = False
            print("Centroids or Cluster are equals")
        else:
            oldc1 = c1
            oldc2 = c2
            oldc3 = c3
            oldcluster1 = cluster1
            oldcluster2 = cluster2
            oldcluster3 = cluster3
            if n > 100:
                next = False
    print("End Simulation: ")

print("Getting the final centroids:")
if type == 1:
    print("\n Using TriUL_SIM\n")
    distance1 = {}
    distance2 = {}
    distance3 = {}
    for em in cluster1:
        for k, v in dictionary1.items():
            if np.array_equal(em, v):
                distance1[k] = TriUL_sim(em, c1)

    for em in cluster2:
        for k, v in dictionary1.items():
            if np.array_equal(em, v):
                distance2[k] = TriUL_sim(em, c2)

    for em in cluster3:
        for k, v in dictionary1.items():
            if np.array_equal(em, v):
                distance3[k] = TriUL_sim(em, c3)

if type == 2:
    print("\n Using Frobenius_Distance\n")
    distance1 = {}
    distance2 = {}
    distance3 = {}
    i = 0
    for em in cluster1:
        for k, v in dictionary1.items():
            if np.array_equal(em, v):
                distance1[k] = Frobenius_Distance(em,c1)

    for em in cluster2:
        for k, v in dictionary1.items():
            if np.array_equal(em, v):
                distance2[k] = Frobenius_Distance(em, c2)

    for em in cluster3:
        for k, v in dictionary1.items():
            if np.array_equal(em, v):
                distance3[k] = Frobenius_Distance(em, c3)



if distance1:
    minkey1 = min(distance1, key=distance1.get)
    print("\nFinal Centroid 1: \n", minkey1)


if distance2:
    minkey2 = min(distance2, key=distance2.get)
    print("\nFinal Centroid 2: \n", minkey2)

if distance3:
    minkey3 = min(distance3, key=distance3.get)
    print("\nFinal Centroid 3: \n", minkey3)


f1 = ""
f2 = ""
f3 = ""
print("\nWriting Files: \n")
for w in sorted(distance1, key=distance1.get):
  f1 += str(distance1[w]) + " , " + w + "\n"
for w in sorted(distance2, key=distance2.get):
  f2 += str(distance2[w]) + " , " + w + "\n"
for w in sorted(distance3, key=distance3.get):
  f3 += str(distance3[w]) + " , " + w + "\n"

if type == 1:
    file1 = open('../test/output/cluster1_TriUL.txt', 'w', encoding='utf-8')
    file1.write(f1)
    file1.close()
    file2 = open('../test/output/cluster2_TriUL.txt', 'w', encoding='utf-8')
    file2.write(f2)
    file2.close()
    file3 = open('../test/output/cluster3_TriUL.txt', 'w', encoding='utf-8')
    file3.write(f3)
    file3.close()

if type ==2:
    file1 = open('../test/output/cluster1_Frobenius.txt', 'w', encoding='utf-8')
    file1.write(f1)
    file1.close()
    file2 = open('../test/output/cluster2_Frobenius', 'w', encoding='utf-8')
    file2.write(f2)
    file2.close()
    file3 = open('../test/output/cluster3_Frobenius', 'w', encoding='utf-8')
    file3.write(f3)
    file3.close()