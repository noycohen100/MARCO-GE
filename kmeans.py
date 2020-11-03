from sklearn.cluster import KMeans

def get_Kmeans(dataframe, K, max_iter):
    kmeans = KMeans(n_clusters=K, max_iter=max_iter)
    kmeans.fit(dataframe)

    return kmeans.labels_
