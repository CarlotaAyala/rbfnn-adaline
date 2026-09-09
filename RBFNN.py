import numpy as np
from sklearn.cluster import KMeans

class RBFNN:
    def __init__(self, numberInputs, numberHideUnits, numberOutputs, eta, output_estimator_class, epochs=500):
        self.numberInputs = numberInputs
        self.numberOutputs = numberOutputs
        self.numberHideUnits = numberHideUnits
        self.eta = eta
        self.epochs = epochs
        self.output_estimator_class = output_estimator_class
        self.centroids = None
        self.variances = None
        self.output_model = None

    def _sum_squared(self, a, b):
        return np.sum((a - b) ** 2)

    def _calculate_variances(self, X, centroids, labels):
        dists = []
        for i, x in enumerate(X):
            c = centroids[labels[i]]
            dists.append(np.linalg.norm(x - c) ** 2)
        sigma2 = np.mean(dists)
        return sigma2

    def _calculate_G(self, X, centroids, sigma2):
        G = np.zeros((X.shape[0], centroids.shape[0]))
        for i, x in enumerate(X):
            for j, c in enumerate(centroids):
                G[i, j] = np.exp(-self._sum_squared(x, c) / (2 * sigma2))
        return G

    def fit(self, X, y):
        kmeans = KMeans(n_clusters=self.numberHideUnits, n_init=10)
        kmeans.fit(X)
        self.centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        self.variances = self._calculate_variances(X, self.centroids, labels)
        G = self._calculate_G(X, self.centroids, self.variances)
        self.output_model = self.output_estimator_class(eta=self.eta, epochs=self.epochs)
        self.output_model.fit(G, y)

    def predict(self, X):
        G = self._calculate_G(X, self.centroids, self.variances)
        return self.output_model.predict(G)
