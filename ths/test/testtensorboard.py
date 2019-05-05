## Importing required Libraries
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import csv

def read_file(size, labeled):
    data = []
    if labeled:
        y = []
        try:
            datafile = open("data/data" + size + "_labeled.csv", "r", encoding='utf-8')
            with datafile as f:
                for row in csv.reader(f):
                    newline = " ".join(row[0].split())
                    data.append(newline)
                    y.append(int(row[1]))
        except Exception as e:
            print(e)
        return data, np.array(y)
    else:
        try:
            datafile = open("data/data" + size + ".txt", "r", encoding='utf-8')
            with datafile as f:
                for line in f:
                    newline = " ".join(line.split())
                    data.append(newline)
        except Exception as e:
            print(e)
        return data

def get_y(size):
    data, y = read_file(size, 1)
    return y

## Path to save the embedding and checkpoints generated
LOG_DIR = '../test/output/log-1/'
## Load data
#outfile = '../test/output/embedded_15k.npy'
outfile = '../test/output/encoded_data_full_gru.npy'
file = np.load(outfile)
#file_2d = file.reshape((file.shape[0], -1))
#df = pd.DataFrame(np.atleast_2d(file_2d))
df = pd.DataFrame(np.atleast_2d(file))

# Metadata consists your labels. Metadata helps us visualize(color) different clusters that form t-SNE
#metadata = os.path.join(LOG_DIR, 'df_labels.tsv')
y = get_y("15k")
#metadata = pd.DataFrame([y])
path_metadata =  os.path.join(LOG_DIR,'metadata.tsv')

# Generating PCA and
pca = PCA(n_components=4, random_state = 123, svd_solver = 'auto')
df_pca = pd.DataFrame(pca.fit_transform(df))
df_pca = df_pca.values
## TensorFlow Variable from data
tf_data = tf.Variable(df_pca)
with open(path_metadata,'w') as f:
    f.write("Index\tLabel\n")
    for index,label in enumerate(y):
        f.write("%d\t%d\n" % (index,label))

## Running TensorFlow Session
with tf.Session() as sess:
    saver = tf.train.Saver([tf_data])
    sess.run(tf_data.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'tf_data.ckpt'))
    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = tf_data.name
    # Link this tensor to its metadata(Labels) file
    embedding.metadata_path = '../log-1/metadata.tsv'
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

    # OPEN THE TERMINAL AND RUN THE NEXT COMMAND: UPDATE PATH
    #tensorboard --logdir = C:\Users\dvill\Documents\Research\DetectDiseaseTHS\ths\test\output\log - 1 --port = 6006