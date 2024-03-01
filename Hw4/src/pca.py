import numpy as np
import matplotlib.pyplot as plt

"""
Implementation of Principal Component Analysis.
"""
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean: np.ndarray = None
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        #TODO: 10%
        # Use existing method to calculate covariance matrix 
        # and its eigenvalues and eigenvectors
        self.mean = np.mean(X, axis=0)
        convariance: np.ndarray = np.cov(X - self.mean, rowvar=False)

        eigenvaules, eigenvectors = np.linalg.eigh(convariance)
        sorted_indices = np.argsort(eigenvaules)[::-1]
        self.components = eigenvectors[:, sorted_indices][:, 0:self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 2%
        return (X - self.mean) @ self.components

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        #TODO: 2%
        return self.transform(X) @ self.components.T + self.mean
