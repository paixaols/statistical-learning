import numpy as np
import pandas as pd

from ..metrics import distance


class KMeans(object):
    def __init__(self, n_clusters, max_iter=100, conv_tol=1e-5, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.conv_tol = conv_tol
        self.random_state = random_state
        self.X = None
        self.labels = None
        self.centers = None
        self.stopping_criterion = None
        self.is_fit = False

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = np.array(X)
        self.X = X.copy()

        # Inicializar os centroides como pontos aleatórios do próprio conjunto
        # de dados
        np.random.seed(self.random_state)
        centers_index = np.random.choice(
            range(len(X)),
            size=self.n_clusters,
            replace=False
        )
        centers = X[centers_index].copy()

        iter_count = 1
        while True:
            clusters = {}
            for p in X:
                # Para cada ponto, calcular distâncias a todos os centroides
                d = distance.array_point(centers, p, metric='euclidean')

                # Atribuir cada ponto ao centroide mais próximo
                closest = d.argmin()
                if clusters.get(closest) is not None:
                    clusters[closest].append(p)
                else:
                    clusters[closest] = [p]

            # Atualizar a posição dos centroides
            new_centers = np.zeros(centers.shape)
            for c in list(clusters.keys()):
                points = np.array(clusters[c])
                new_centers[c] = points.mean(axis=0)

            # Avaliar critério de parada
            if iter_count >= self.max_iter:
                self.stopping_criterion = 'MAX ITERATIONS'
                break
            iter_count += 1

            centers_displacement = []
            for i in range(len(centers)):
                d = distance.array_point(new_centers[[i],:], centers[i])
                centers_displacement.append(d[0])
            if max(centers_displacement) < self.conv_tol:
                self.stopping_criterion = 'CONVERGENCE'
                break

            centers = new_centers.copy()
        self.centers = new_centers.copy()

        labels = []
        for p in X:
            d = distance.array_point(self.centers, p, metric='euclidean')
            closest = d.argmin()
            labels.append(closest)
        self.labels = np.array(labels)

        self.is_fit = True

        # Avaliação da métrica WSS - Within Cluster Sum of Squares
        self.wss = {}
        self.wss_total = 0
        for i in range(self.centers.shape[0]):
            center = self.centers[i]
            cluster = clusters[i]
            d = distance.array_point(cluster, center)
            sum_squares = np.sum(d**2)
            self.wss[i] = sum_squares
            self.wss_total += sum_squares

        return self.labels