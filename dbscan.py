
from sklearn.cluster import DBSCAN

def get_dbscan(dataframe, initial_eps=0.01):
    eps = initial_eps
    model = DBSCAN(eps=eps, min_samples=4)
    model.fit(dataframe)
    unlabeled = (model.labels_ == -1).astype(int).sum()
    while unlabeled > dataframe.shape[0] * 0.1:
        eps *= 1.1
        model = DBSCAN(eps=eps, min_samples=4)
        model.fit(dataframe)
        unlabeled = (model.labels_ == -1).astype(int).sum()
    return model.labels_