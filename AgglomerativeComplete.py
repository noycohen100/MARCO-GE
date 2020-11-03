from sklearn.cluster import AgglomerativeClustering

def get_AgglomerativeComplete(dataframe, K):
    AgglomerativeComplete = AgglomerativeClustering(n_clusters=K, linkage="complete")
    AgglomerativeComplete.fit(dataframe)

    return AgglomerativeComplete.labels_
