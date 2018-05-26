# import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import numpy as np
from src.clustering.spectral_clustering import SpectralClustering
from src.clustering.ward import Ward
from src.graph.graph_generator import GraphGenerator
from src.kernel.kernel import get_all_kernels, Forest
from src.kernel.transformation import get_all_transformations, OneThirdTransform, SqrtTransformation, \
     NoTransformation, SquareTransformation, ArcTanTransformation, get_new_transformations
from src.kernel.transformed_kernel import TransformedKernel

generator = GraphGenerator(100, 4, 0.2, 0.05)
graph = generator.generate_graph()
nodes = graph.nodes
edges = graph.edges
transformations = get_new_transformations()
transformations.append(NoTransformation)
for kernel_class in get_all_kernels():
    for transformation in transformations:
        scores = []
        kernel = TransformedKernel(kernel_class(edges), transformation())
        for K in kernel.get_Ks():
            K = np.nan_to_num(K)
            prediction = Ward(4).fit_predict(K)
            score = adjusted_rand_score(nodes, prediction)
            scores.append(score)
        print(kernel_class.name, transformation.name, max(scores))
        # plt.plot(kernel_class.default_params, scores)
