from sklearn.cluster import MiniBatchKMeans

def get_MiniBatchKmeans(dataframe, K, max_iter):
    kmeans = MiniBatchKMeans(n_clusters=K)

    kmeans.fit(dataframe)
    return kmeans.labels_
