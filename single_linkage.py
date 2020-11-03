from scipy.cluster.hierarchy import single, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics.cluster import silhouette_score
import numpy as np

def get_single_linkage(dataframe):
    dists = pdist(dataframe)
    Z = single(dists)
    best_score = (-1, 2)
    last_score = -1
    non_improving_iter = 0
    k = 2
    while non_improving_iter < 10:
        labels = fcluster(Z, k, criterion='maxclust')

        if len(np.unique(labels)) > 1:
            res = silhouette_score(dataframe, labels)
        
            if res > last_score:
                non_improving_iter = 0
            else:
                non_improving_iter+=1

            if res > best_score[0]:
                best_score = (res, k, labels)
        
            last_score = res
        k+=1

    return best_score[2]