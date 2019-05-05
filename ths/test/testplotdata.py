# That's an impressive list of imports.
import numpy as np
import csv
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities, _kl_divergence)
#from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy


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


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 3))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(3):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


if __name__ == "__main__":
    type = 2 # 1:embedded data, 2: encoded data
    file_size = ('15k', 72)  # options: ('15k', 72)
    if type == 1:
        outfile = '../test/output/embedded_' + file_size[0] + '.npy'
        file = np.load(outfile)
        y = get_y(file_size[0])

    if type == 2:
        outfile = '../test/output/encoded_data_dense_t1.npy'
        file = np.load(outfile)
        y = get_y(file_size[0])

    # select first 2000 rows
    y_limit = y[10000:]
    file_2d = file.reshape((file.shape[0], -1))
    file_2d_limit = file_2d[10000:, ]

    x_final = np.vstack([file_2d_limit[y_limit == i] for i in range(3)])
    y_final = np.hstack([y_limit[y_limit == i] for i in range(3)])
    tweet_proj = TSNE(random_state=RS).fit_transform(x_final)

    scatter(tweet_proj, y_final)
    if type == 1:
        plt.savefig("../test/output/cluster_tsne_embedded.png", dpi=120)
    if type == 2:
        plt.savefig('../test/output/cluster_tsne_encoded_dense_t1.png', dpi=120)