import numpy as np
from abc import abstractmethod, ABC


class Scaler(ABC):
    def __init__(self, A):
        self.A = A

    @abstractmethod
    def scale(self, param):
        pass


class Rho(Scaler):  # plain walk
    def __init__(self, A):
        super().__init__(A)
        eigvals = np.linalg.eigvals(self.A)
        self.rho = np.max(eigvals)

    def scale(self, param):
        return param / self.rho


class Fraction(Scaler):  # communicability
    def scale(self, param):
        return 0.5 * param / (1.0 - param)