from abc import abstractmethod, ABC

from scipy.linalg import expm

from src.kernel.scaler import Scaler, Fraction, Rho
import numpy as np


class Kernel(ABC):
    default_params = np.arange(0.05, 1, 0.025)

    def __init__(self, A):
        self.A = A
        self.n = self.A.shape[0]

    def get_params(self):
        return self.scaler.scale(Kernel.default_params)

    def get_Ks(self):
        Ks = []
        for param in self.get_params():
            Ks.append(self.get_K(param))
        return Ks

    def get_D(self):
        return np.diag(np.sum(self.A, axis=0))

    def get_L(self):
        return self.get_D() - self.A

    @abstractmethod
    def get_K(self, param):
        pass

    @property
    @abstractmethod
    def scaler(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass


class PlainWalk(Kernel):
    name = 'PlainWalk'

    def get_K(self, param):
        return np.linalg.pinv(np.eye(self.n) - param * self.A)

    @property
    def scaler(self):
        return Rho(self.A)


class Communicability(Kernel):
    name = 'Communicability'

    def get_K(self, param):
        return expm(param * self.A)

    @property
    def scaler(self):
        return Fraction(self.A)


class Forest(Kernel):
    name = 'Forest'

    def get_K(self, param):
        return np.linalg.pinv(np.eye(self.n) + param * self.get_L())

    @property
    def scaler(self):
        return Fraction(self.A)


class Heat(Kernel):
    name = 'Heat'

    def get_K(self, param):
        return expm(-param * self.get_L())

    @property
    def scaler(self):
        return Fraction(self.A)

def get_all_kernels():
    return [PlainWalk, Communicability, Forest, Heat]
