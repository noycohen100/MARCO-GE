from mst_clustering import MSTClustering

def get_mst(dataframe):
    model = MSTClustering(cutoff_scale=2)

    model.fit(dataframe)
    return model.labels_