import numpy as np
import pandas as pd
import os
import math
def find_best_algorithm(clustering_measure):
    col_drop = []
    algorithm_performance_best_index = {}
    path ='Results/'+ clustering_measure+"/"
    hist_algo = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for dataset in sorted(os.listdir(path), key=lambda s: s.lower()):
        dataset_rankings_with_name = pd.read_csv(path + dataset, header=0)
        dataset_name = dataset.split('\\')[-1].split('.')[0]
        dataset_rankings = dataset_rankings_with_name.drop(columns=['dataset'], axis=1)
        dataset_rankings = dataset_rankings.drop(columns=col_drop, axis=1).values
        min_ranking = np.argmin(dataset_rankings)
        hist_algo[min_ranking] = hist_algo[min_ranking] + 1
        algorithm_performance_best_index[dataset_name] = min_ranking
    label_file_name= "labels_" +clustering_measure+".txt"
    f = open(label_file_name,"w")
    for tuple in algorithm_performance_best_index.items():
        f.write(tuple[0]+";"+str(tuple[1]) +"\n")
    print(hist_algo)
    return label_file_name

def union_algorithms_performance(clustering_measure):
    cols= ['dbscan', 'mst', 'SL', 'eac', 'khmeans', 'kmeans', 'psc','aa','ac','kmb','aw','kkm' ,'fuzzy','dataset']
    algorithm_performance_index=pd.DataFrame(columns= cols)
    if clustering_measure == "average":
        path ='Results/'+ clustering_measure+"/"
    else:
        path ='Results/' + clustering_measure + "_full/"

    training_file_name = "training_" + clustering_measure+".csv"
    for dataset in sorted(os.listdir(path), key=lambda s: s.lower()):
        dataset_name = dataset.split('\\')[-1].split('.')[0]
        dataset_algorithms_perforamce = pd.read_csv(path+dataset, header=0)
        dataset_algorithms_perforamce['dataset'] =dataset_name
        dataset_algorithms_perforamce = dataset_algorithms_perforamce.replace(math.inf, 1000)
        algorithm_performance_index = algorithm_performance_index.append(dataset_algorithms_perforamce)
    # df = pd.DataFrame.from_records(algorithm_performance_index, columns=cols)
    algorithm_performance_index.to_csv(training_file_name, columns=cols)
    return training_file_name
