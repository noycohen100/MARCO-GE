import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd

class DatasetValues():

    def __init__(self, dataframe: pd.DataFrame, cluster_labels):

        self.data = dataframe
        self.cluster_labels = cluster_labels
        self.unique_labels = np.unique(cluster_labels)
        self.unique_labels = np.delete(self.unique_labels, np.where(self.unique_labels == -1), axis=0)
        
        self.intra = []

        # Computing intra-cluster measures
        for k in self.unique_labels:
            points = self.data.iloc[np.where(self.cluster_labels==k)]
            self.intra.append(max(euclidean_distances(points).reshape(-1)))
        
        self.K = len(self.unique_labels)
        # Computing inter-cluster
        inter = np.zeros((self.K, self.K))
        self.inter = []
        for i in range(self.K):
            for j in range(self.K):
                if self.unique_labels[i]!=self.unique_labels[j]:
                    points_A = self.data.iloc[np.where(self.cluster_labels==self.unique_labels[i])]
                    points_B = self.data.iloc[np.where(self.cluster_labels==self.unique_labels[j])]

                    dist = sum(euclidean_distances(points_A, points_B).reshape(-1)) / (points_A.shape[0] * points_B.shape[0])
                    inter[i][j] = dist
        
        for i in range(self.K):
            for j in range(i+1, self.K):
                self.inter.append(inter[i][j] + inter[j][i])

        # Calculating same, diff pairs
        self.same_pairs = 0
        self.diff_pairs = 0
        self.mean_same = 0
        self.mean_diff = 0
        var_total = 0
        mean_total = 0

        for i in range(self.data.shape[0]):
            for j in range(i+1, self.data.shape[0]):

                d = np.linalg.norm(self.data.values[i] - self.data.values[j])

                if self.cluster_labels[i] == self.cluster_labels[j]:
                    self.same_pairs += 1
                    self.mean_same += ((d - self.mean_same) / self.same_pairs)
                else:
                    self.diff_pairs += 1
                    self.mean_diff += ((d - self.mean_diff) / self.diff_pairs)
                
                var_total += pow(d-mean_total, 2) - (pow(d-mean_total, 2)/(self.same_pairs + self.diff_pairs))
                mean_total += ((d - mean_total) / (self.same_pairs + self.diff_pairs))
        
        var_total = var_total / (self.same_pairs+self.diff_pairs-1)
        self.std_total = np.sqrt(var_total)
        




        