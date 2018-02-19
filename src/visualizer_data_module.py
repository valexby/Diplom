from collections import Counter


import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from mvpa2.suite import *

class VisualizeDataModule:

    def dimension_reduction(self, data, n_component, reduction_method):
        if reduction_method == 't_sne':
            model = TSNE(n_components=n_component, random_state=0)
        elif reduction_method == 'pca':
            model = PCA(n_components=n_component)
        else:
            raise  ValueError("non valid reduction algorithm")
        X_tsne = model.fit_transform(data)
        X_tsne -= X_tsne.min(axis=0)
        X_tsne /= X_tsne.max(axis=0)
        np.set_printoptions(suppress=True)
        return X_tsne

    def plot_2d(self, data, labels, genre_list, show=True, reduction_method='t_sne'):
        X_tsne = self.dimension_reduction(data, 2, reduction_method=reduction_method)
        plt.figure()

        start = 0
        colors = cm.rainbow(np.linspace(0, 1, len(genre_list)))
        c = []
        for color, size, genre in zip(colors,
                                      Counter(labels).values()[:len(genre_list)],
                                      genre_list):
            xs = map(lambda x: x[0], X_tsne)[start: start + size]
            ys = map(lambda x: x[1], X_tsne)[start: start + size]
            start += size
            c.append(plt.scatter(xs, ys, c=color, label=genre))

        plt.legend(c,
                   genre_list,
                   scatterpoints=1,
                   loc='lower left',
                   ncol=3,
                   fontsize=8)
        plt.savefig('2d.pdf')
        if show:
            plt.show()

    def plot_3d(self, data, labels, genre_list, show=True, reduction_method='t_sne'):

        X_tsne = self.dimension_reduction(data, 3, reduction_method=reduction_method)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        start = 0
        colors = cm.rainbow(np.linspace(0, 1, len(genre_list)))
        for color, size, genre in zip(colors,
                                      Counter(labels).values()[:len(genre_list)],
                                      genre_list):
            xs = map(lambda x: x[0], X_tsne)[start: start + size]
            ys = map(lambda x: x[1], X_tsne)[start: start + size]
            zs = map(lambda x: x[2], X_tsne)[start: start + size]
            start += size
            ax.scatter(xs, ys, zs, c=color, label=genre)

        plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=16, bbox_to_anchor=(0, 0))
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.savefig('3d.pdf')
        if show:
            plt.show()

    def plot_som(self, data, labels, genre_list):
        model = SimpleSOMMapper((1000, 200), 10, learning_rate=0.05)
        data = np.array(data)
        model.train(data)
        plt.imshow(model.K, origin='lower')
        mapped = model(data)

        pl.title('Color SOM')
        # SOM's kshape is (rows x columns), while matplotlib wants (X x Y)
        for i, m in enumerate(mapped):
            pl.text(m[1], m[0], genre_list[i], ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.5, lw=0))
        plt.show()