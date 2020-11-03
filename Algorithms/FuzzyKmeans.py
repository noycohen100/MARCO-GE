from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans

def get_FuzzyKmeans(dataframe,K):
    mdl = FuzzyKMeans(k=K)
    mdl.fit(dataframe)
    return mdl.labels_
