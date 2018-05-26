from abc import abstractmethod, ABC

import numpy as np


class Transformation(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def transform(self, K):
        pass


class NoTransformation(Transformation):
    name = 'NoTransformation'

    def transform(self, K):
        return K


class LogTransformation(Transformation):
    name = 'LogTransformation'

    def transform(self, K):
        K = np.nan_to_num(K)
        return np.log(K)


class SquareTransformation(Transformation):
    name = 'SquareTransformation'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = pow(K[i, j], 2)
        return K


class OneThirdTransform(Transformation):
    name = 'OneThirdTransform'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = pow(K[i, j], 1.0 / 3.0)
        return K


class OneFourthTransform(Transformation):
    name = 'OneFourthTransform'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = pow(K[i, j], 1.0 / 4.0)
        return K


class OneFifthTransform(Transformation):
    name = 'OneFifthTransform'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = pow(K[i, j], 1.0 / 5.0)
        return K


class OneSixthTransform(Transformation):
    name = 'OneSixthTransform'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = pow(K[i, j], 1.0 / 6.0)
        return K


class OneSeventhTransform(Transformation):
    name = 'OneSeventhTransform'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = pow(K[i, j], 1.0 / 7.0)
        return K


class OneEigthTransform(Transformation):
    name = 'OneEigthTransform'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = pow(K[i, j], 1.0 / 8.0)
        return K


class OneNinthTransform(Transformation):
    name = 'OneNinthTransform'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = pow(K[i, j], 1.0 / 9.0)
        return K


class OneTenthTransform(Transformation):
    name = 'OneTenthTransform'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = pow(K[i, j], 1.0 / 10.0)
        return K


class WTFTransform(Transformation):
    name = 'WTFTransform'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = pow(K[i, j], 1.0 / 15.0)
        return K


class WTF2Transform(Transformation):
    name = 'WTF2Transform'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = pow(K[i, j], 1.0 / 35.0)
        return K


class WTF3Transform(Transformation):
    name = 'WTF3Transform'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = pow(K[i, j], 1.0 / 200.0)
        return K


class WTF4Transform(Transformation):
    name = 'WTF4Transform'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = pow(K[i, j], 1.0 / 1000.0)
        return K


class SqrtTransformation(Transformation):
    name = 'SqrtTransformation'

    def transform(self, K):
        return np.sqrt(K)


class ExpTransformation(Transformation):
    name = 'ExpTransformation'

    def transform(self, K):
        return np.exp(K)


class SigmoidTransformation(Transformation):
    name = 'SigmoidTransformation'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = sigmoid(K[i, j])
        return K


class ArcTanTransformation(Transformation):
    name = 'ArcTanTransformation'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = np.arctan(K[i, j])
        return K


class ReluTransformation(Transformation):
    name = 'ReluTansformation'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                if K[i, j] <= 0:
                    K[i, j] = 0

        return K


class TanHTransformation(Transformation):
    name = 'TanHTransformation'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = np.tanh(K[i, j])

        return K


class ISRUTransformation(Transformation):
    name = 'ISRUTransformation'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = (K[i, j] / np.sqrt((1 + 1 * K[i, j] * K[i, j])))  # second 1 may be param

        return K


class SoftsignTransformation(Transformation):
    name = 'SoftsignTransformation'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = (K[i, j] / (1 + np.abs(K[i, j])))

        return K


class SoftplusTransformation(Transformation):
    name = 'SoftplusTransformation'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = np.log(1 + np.exp(K[i, j]))

        return K


class SiLUTransformation(Transformation):
    name = 'SiLUTransformation'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = K[i, j] * sigmoid(K[i, j])

        return K


class GaussianTransformation(Transformation):
    name = 'GaussianTransformation'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                K[i, j] = np.exp(-1 * K[i, j] * K[i, j])  # second 1 may be param

        return K

class ReluInvertedTransformation(Transformation):
    name = 'ReluInvertedTransformation'

    def transform(self, K):
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                if K[i, j] >= 100:
                    K[i, j] = 100

        return K



def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-1.0 * x))
    return y


def get_all_transformations():
    return [NoTransformation, LogTransformation, SquareTransformation,
            SqrtTransformation, SigmoidTransformation, OneThirdTransform, OneFourthTransform,
            OneFifthTransform, OneTenthTransform,
            ArcTanTransformation, ReluTransformation, TanHTransformation, ISRUTransformation, SoftsignTransformation,
            SoftplusTransformation, SiLUTransformation
            ]


def get_new_transformations():
    return [ArcTanTransformation, ReluTransformation, TanHTransformation, ISRUTransformation, SoftsignTransformation,
            SoftplusTransformation, SiLUTransformation]

def get_wtf_transformations():
    return [WTFTransform, WTF2Transform, WTF3Transform, WTF4Transform]
