from sklearn.cluster import AgglomerativeClustering

def get_AgglomerativeAverage(dataframe, K):
    AgglomerativeAverage = AgglomerativeClustering(n_clusters=K, linkage="average")
    AgglomerativeAverage.fit(dataframe)

    return AgglomerativeAverage.labels_
