from sklearn.cluster import AgglomerativeClustering

def get_AgglomerativeWard(dataframe, K):
    AgglomerativeWard = AgglomerativeClustering(n_clusters=K, linkage="ward")
    AgglomerativeWard.fit(dataframe)

    return AgglomerativeWard.labels_
