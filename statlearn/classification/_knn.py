from .base import BaseClassifier
from ..metrics import distance


class KNNClassifier(BaseClassifier):
    def __init__(self, n_neighbors=5, metric='euclidean'):
        super().__init__(
            n_neighbors=n_neighbors,
            metric=metric
        )

    def predict(self, X):
        if not self.is_fit:
            raise Exception('Call "fit" method before "predict" method')

        prediction = []
        for i in range(X.shape[0]):
            p = X.iloc[i]

            # Calcular a distância entre o ponto sem rótulo e todos os pontos com rótulo
            d = distance.array_point(self.X_train, p)

            # Encontrar os K pontos rotulados mais próximos
            aux = self.X_train.copy()
            aux['dist'] = d
            aux.sort_values('dist', inplace=True)

            nearest_samples_index = aux.iloc[:self.n_neighbors].index
            nearest_labels = self.y_train[nearest_samples_index]

            # Contar os pontos de cada classe entre os vizinhos próximos (votação)
            # TODO: tratar casos de empate
            predicted_label = nearest_labels.value_counts().index[0]
            prediction.append(predicted_label)
        return prediction