from sklearn.metrics.cluster import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from Utils.dataset_values import DatasetValues
from sklearn.metrics.pairwise import euclidean_distances
import openensembles as op
from Metrics.indices import internalIndex
import numpy as np

def fr_score(dataset_values: DatasetValues):
    """
    Friedman, J. H.; Rafsky, L. C. (1979). Multivariate generalizations of
    the waldwolfowitz and smirnov two-sample tests. Annals of Statistics,
    v.7, n.4, p.697�717.

    The objective is minimize value [0, +1]
    """
    return 0

def hkk_score(dataset_values: DatasetValues):
    """
    Handl, J.; Knowles, J.; Kell, D. B. (2005). Computational cluster validation
    in post-genomic data analysis. Bioinformatics, v.21, n.15, p.3201�3212.

    The objective is minimize value [0, +Inf]
    """

    if dataset_values.K == 1:
        
        return np.inf
    
    # get the smallest cluster
    L = min(np.unique(dataset_values.cluster_labels, return_counts=True)[1])
    L = int(np.ceil(np.sqrt(L)))
    diffCluster = 0
    value = np.zeros(dataset_values.data.shape[0])
    for i in range(dataset_values.data.shape[0]):
        neighbors = np.argsort(euclidean_distances(dataset_values.data.values[i].reshape(1,-1), dataset_values.data.values).reshape(-1))
        neighbors = neighbors[1:L+1]

        for j in range(L):
            if dataset_values.cluster_labels[i] != dataset_values.cluster_labels[neighbors[j]]:
                diffCluster = 1
            else:
                diffCluster = 0
            value[i] += diffCluster / (j+1)
    
    return sum(value)

def hl_score(dataset_values: DatasetValues):
    """Hubert, L. J.; Levin, J. R. (1976). A general statistical framework for
       assessing categorical clustering in free recall. Psychological Bulletin,
      v.83, n.6, p.1072�1080.
       The objective is minimize value [0, +1]"""

    if dataset_values.K == 1:
        return 1

    same_min = []
    same_max = []
    for i in range(dataset_values.data.shape[0] - 1):
        for j in range(i+1, dataset_values.data.shape[0]):
            d = np.linalg.norm(dataset_values.data.values[i].reshape(1,-1) - dataset_values.data.values[j].reshape(1,-1))
            same_min.append(d)
            same_max.append(d)
        if len(same_min) > dataset_values.same_pairs:
            same_min = sorted(same_min, reverse=False)
            same_max = sorted(same_max, reverse=True)
            same_min = same_min[:dataset_values.same_pairs]
            same_max = same_max[:dataset_values.same_pairs]
    
    S = dataset_values.mean_same * dataset_values.same_pairs
    Smin = sum(same_min)
    Smax = sum(same_max)

    return (S - Smin) / (Smax - Smin)
    

def mc_score(dataset_values: DatasetValues):
    """
   Milligan, G.W.; Cooper, M.C. (1985). An examination of procedures for
   determing the number of clusters in a data set. Psychometrika, v.1.

   The objective is maximize value [-1, +1]
    """
    if dataset_values.K == 1:
        return -1
    
    t = dataset_values.same_pairs + dataset_values.diff_pairs
    Wd = dataset_values.same_pairs
    Bd = dataset_values.diff_pairs

    meanDb = dataset_values.mean_diff
    meanDw = dataset_values.mean_same
    Sd = dataset_values.std_total

    return (meanDb-meanDw) * np.sqrt((Wd*Bd) / (t*t)) / Sd


def bp_score(dataset_values: DatasetValues):
    """
    Bezdek, J. C.; Pal, N. R. (1998b). Some new indexes of cluster validity.
    IEEE Transactions on Systems, Man, and Cybernetics, Part B, v.28, n.3,
    p.301�315

    The objective is maximize value [0, +Inf]
    """
    if dataset_values.K == 1:
        return 0

    return np.mean(dataset_values.inter)


def gk_score(dataset_values: DatasetValues):
    """
   Baker, F.B.; Hubert, L.J. (1975). Measuring the power of hierarchical
   cluster analysis. Journal of the American Statistical Associations,
   v.40, n.349, p.31-38

   The objective is maximize value [-1, +1]
    """
    if dataset_values.K == 1:
        return -1

    same = []
    diff = []

    for i in range(len(dataset_values.cluster_labels) - 1):
        for j in range(i+1, len(dataset_values.cluster_labels)):
            same.append((i, j)) if dataset_values.cluster_labels[i] == dataset_values.cluster_labels[j] else diff.append((i,j))
    
    concordant = 0

    for i in range(len(same)):
        for j in range(len(diff)):
            same_d = np.linalg.norm(dataset_values.data.values[same[i][0]].reshape(1,-1) - dataset_values.data.values[same[i][1]].reshape(1,-1))
            diff_d = np.linalg.norm(dataset_values.data.values[diff[j][0]].reshape(1,-1) - dataset_values.data.values[diff[j][1]].reshape(1,-1))

            if same_d < diff_d:
                concordant+=1
    
    total = len(same) * len(diff)
    discordant = total - concordant

    return (concordant - discordant) / total
    

def dunn_score(dataset_values: DatasetValues):
    """
   Dunn, J. (1973). A fuzzy relative of the isodata process and its use in
   detecting compact well-separated clusters. J. Cybernet, v.3, n.3, p.32-57

   The objective is maximize value [0, +Inf]
    """
    if dataset_values.K == 1:
       return 0

    return min(dataset_values.inter) / max(dataset_values.intra)

def calinski_harabasz(dataset_values:DatasetValues):
    """Calinski, T.; Harabasz, J. (1974). A dendrite method for cluster analysis.
    Communications in Statistics - Theory and Methods, v.3, n.1, p.1�27.
    The objective is maximize value [0, +Inf]"""

    if dataset_values.K == 1:
        return 0
        
    return calinski_harabasz_score(dataset_values.data, dataset_values.cluster_labels)

def silhouettes(dataset_values: DatasetValues):
    """
   Rousseeuw, P. (1987). Silhouettes: a graphical aid to the interpretation
   and validation of cluster analysis. J. Comput. Appl. Math., v.20, n.1,
   p.53�65.

   The objective is maximize value [-1, +1]
    """

    if dataset_values.K == 1:
        return -1

    return silhouette_score(dataset_values.data, dataset_values.cluster_labels)

def davies_bouldin(dataset_values: DatasetValues):
    """
   Davies, D. L.; Bouldin, D. W. (1979). A cluster separation measure. IEEE
   Trans. Pattern Anal. Mach. Intell., v.1, n.2, p.224�227.

   The objective is minimize value [0, +Inf]
    """
    if dataset_values.K == 1:
        return np.inf

    return davies_bouldin_score(dataset_values.data, dataset_values.cluster_labels)




def XieBeni_score(dataset_values: DatasetValues):
    data_array = dataset_values.data.values
    xiB = internalIndex(len(dataset_values.unique_labels))
    xiB_score = xiB.xie_benie(data_array, dataset_values.cluster_labels)
    return xiB_score

def Scat_score(dataset_values: DatasetValues):
    data_array = dataset_values.data.values
    s = internalIndex(len(dataset_values.unique_labels))
    Scat_s= s.Scat(data_array, dataset_values.cluster_labels)
    return Scat_s

def get_internal_measures(values: DatasetValues):

    return {
        "DB": davies_bouldin(values),
        "SIL": silhouettes(values),
        "CH": calinski_harabasz(values),
        "DU": dunn_score(values),
        # "GK": gk_score(values),
        "BP": bp_score(values),
        "MC": mc_score(values),
        "HL": hl_score(values),
        "HKK": hkk_score(values),
        "Xie": XieBeni_score(values),
        "Scat": Scat_score(values),
        # "FR": fr_score(values)
    }