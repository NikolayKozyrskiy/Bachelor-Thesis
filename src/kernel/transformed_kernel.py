class TransformedKernel:
    def __init__(self, kernel, transformation):
        self.kernel = kernel
        self.transformation = transformation

    def get_Ks(self):
        transformed_Ks = []
        for K in self.kernel.get_Ks():
            transformed_Ks.append(self.transformation.transform(K))
        return transformed_Ks

    def get_K(self, param):
        return self.transformation.transform(self.kernel.get_K(param))
