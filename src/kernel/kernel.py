import scipy
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
        self.A = 0.5 * (self.A + self.A.T)
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
        return np.linalg.pinv(np.eye(self.n) - param * self.A).real

    @property
    def scaler(self):
        return Rho(self.A)

class LogPlainWalk(Kernel):
    name = 'LogPlainWalk'

    def get_K(self, param):
        k = np.nan_to_num(np.linalg.pinv(np.eye(self.n) - param * self.A)).real
        return np.nan_to_num(np.log(k))

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

class LogCommunicability(Kernel):
    name = 'LogCommunicability'

    def get_K(self, param):
        k = np.nan_to_num(expm(param * self.A))
        return np.nan_to_num(np.log(k))

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

class LogForest(Kernel):
    name = 'LogForest'

    def get_K(self, param):
        k = np.nan_to_num(np.linalg.pinv(np.eye(self.n) + param * self.get_L()))
        return np.nan_to_num(np.log(k))

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

class LogHeat(Kernel):
    name = 'LogHeat'

    def get_K(self, param):
        k = np.nan_to_num(expm(-param * self.get_L()))
        return np.nan_to_num(np.log(k))

    @property
    def scaler(self):
        return Fraction(self.A)

class SigmoidCommuteTime(Kernel):
    name='SigmoidCommuteTime'

    def get_K(self, param):
        Kct = np.linalg.pinv(self.get_L())
        sigma = Kct.std()
        for i in range(0, Kct.shape[0]):
            for j in range(0, Kct.shape[1]):
                Kct[i, j] = 1.0 / (1.0 + param * np.exp(-1.0 * Kct[i, j])/sigma)
        return np.nan_to_num(Kct)

    @property
    def scaler(self):
        return Fraction(self.A)


class LogSigmoidCommuteTime(Kernel):
    name = 'LogSigmoidCommuteTime'

    def get_K(self, param):
        Kct = np.linalg.pinv(self.get_L()).real
        for i in range(0, Kct.shape[0]):
            for j in range(0, Kct.shape[1]):
                Kct[i, j] = 1.0 / (1.0 + param * np.exp(-1.0 * Kct[i, j]))
        return np.nan_to_num(np.log(Kct))

    @property
    def scaler(self):
        return Fraction(self.A)


class SigmoidCorrectedCommuteTime(Kernel):
    name='SigmoidCorrectedCommuteTime'

    def get_K(self, param):
        H = np.eye(self.n) - np.dot(np.ones(self.n), np.ones(self.n).T)/self.n
        D_inv = np.linalg.pinv(self.get_D())
        for i in range(D_inv.shape[0]):
            for j in range(D_inv.shape[1]):
                D_inv[i, j] = pow(D_inv[i, j], 1./2)
        d = self.A*np.ones(self.n)
        vol = sum((self.A != 0).sum(0))/2
        M = np.dot(np.dot(D_inv, (self.A - np.dot(d, d.T) / vol)), D_inv)
        Kcct = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(H, D_inv), M),
                                            np.linalg.pinv(np.eye(self.n) - M)), M), D_inv), H)
        # Kcct = H * D_inv * M * np.linalg.pinv(np.eye(self.n) - M) * M * D_inv * H
        for i in range(0, Kcct.shape[0]):
            for j in range(0, Kcct.shape[1]):
                Kcct[i, j] = 1.0 / (1.0 + param * np.exp(-1.0 * Kcct[i, j]))
        return Kcct

    @property
    def scaler(self):
        return Fraction(self.A)

def get_all_kernels():
    return [PlainWalk, LogPlainWalk, Communicability, LogCommunicability,
            Forest, LogForest, Heat, LogHeat, SigmoidCommuteTime, LogSigmoidCommuteTime]
