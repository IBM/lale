class UnknownOp:
    def __init__(self, n_neighbors=5, algorithm='auto'):
        self._hyperparams = {
            'n_neighbors': n_neighbors, 'algorithm': algorithm}
    def get_params(self):
        return self._hyperparams
    def fit(self, X, y):
        self._sklearn_model = sklearn.neighbors.KNeighborsClassifier(
            **self._hyperparams)
    def predict(self, X):
        return self._sklearn_model.predict(X)
