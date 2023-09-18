import numpy as np


"""
Implementation of Principal Component Analysis.
"""
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        #TODO: 10%
        # Hint: Use existing method to calculate covariance matrix 
        # and its eigenvalues and eigenvectors
        self.mean: np.ndarray = np.mean(X, axis=0)
        covariance_matrix = np.cov(X - self.mean, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        



    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 2%
        raise NotImplementedError

    def reconstruct(self, X):
        raise NotImplementedError
        #TODO: 2%
