class BaseClassifier(object):
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.is_fit = False

    def fit(self, X, y):
        # TODO: validar dados
        self.X_train = X
        self.y_train = y
        self.is_fit = True