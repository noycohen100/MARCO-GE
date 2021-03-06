import sys
import pandas as pd
from Utils.preprocessing import preprocess
from Algorithms.MiniBatchKmeans import get_MiniBatchKmeans
from Algorithms.AgglomerativeWard import get_AgglomerativeWard
from Algorithms.dbscan import get_dbscan
from Algorithms.KernalKmeans import get_KeranlKMeans
from Algorithms.AgglomerativeAverage import get_AgglomerativeAverage
from Algorithms.khmeans import get_khmeans
from Algorithms.kmeans import get_Kmeans
from Algorithms.mst import get_mst
from Algorithms.FuzzyKmeans import get_FuzzyKmeans
from Algorithms.single_linkage import get_single_linkage
from Algorithms.AgglomerativeComplete import get_AgglomerativeComplete
from Algorithms.eac import get_EAC
from Algorithms.psc import get_psc
from Metrics.metrics import get_internal_measures
from Utils.dataset_values import DatasetValues
import os
from glob import glob
import pickle
import json
import numpy as np
from scipy.stats import rankdata
from multiprocessing import Pool
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run(filename: str, dataframe: pd.DataFrame, algname: str, K=None):

    if algname == "dbscan":
        labels = get_dbscan(dataframe)
    elif algname == "khm":
        labels = get_khmeans(dataframe, K, 30)
    elif algname == "km":
         labels = get_Kmeans(dataframe, K, 50)
    elif algname == "mst":
        labels = get_mst(dataframe)
    elif algname == "SL":
         labels = get_single_linkage(dataframe)
    elif algname == "eac":
        labels = get_EAC(dataframe)
    elif algname == "psc":
        labels = get_psc(dataframe, K)
    elif algname== "al":
        labels = get_AgglomerativeAverage(dataframe, K)
    elif algname == "cl":
        labels = get_AgglomerativeComplete(dataframe,K)
    elif algname == "mbk":
        labels = get_MiniBatchKmeans(dataframe,K,50)
    elif algname == "wl":
        labels = get_AgglomerativeWard(dataframe,K)
    elif algname =="kkm":
        labels =get_KeranlKMeans(dataframe,K)
    elif algname =="fc":
        labels =get_FuzzyKmeans(dataframe,K)
    values = DatasetValues(dataframe, labels)
    measures = get_internal_measures(values)
    # save_path = os.path.join(os.getcwd(), 'Results_Labels', algname, filename.split('/')[-1])
    #
    # # json.dump(measures, open(save_path + "_" + algname + ".json", 'w'))
    # pickle.dump(labels, open(save_path + "_" + algname + ".clusters", 'wb'))

    return measures, labels

#Writing index values before ranking
def saveResultsIndex(dataset_name, index_name, measures, results_dir):
    measures = np.append(measures, dataset_name)
    df = pd.DataFrame(measures).transpose()
    if not os.path.exists(results_dir + '/' + index_name):
        os.makedirs(results_dir + '/' + index_name)
    save_path = os.path.join(os.getcwd(), results_dir, index_name,dataset_name.split('/')[-1])
    #json.dump(measures, open(save_path + ".json", 'w'))
    header = ['dbscan', 'mst', 'SL', 'eac', 'khm', 'km', 'psc', 'al', 'cl', 'mbk', 'wl', 'kkm','fc','dataset']
    df.to_csv(save_path+ ".csv", index=False, header=header)


def get_rankings_per_index_final(measures,dataset_name):
    header =['dbscan', 'mst', 'SL', 'eac', 'khm', 'km', 'psc', 'al','cl','mbk','wl','kkm','fc','dataset']
    min_internal_measures = ["DB", "HL","HKK", "Xie", "Scat"]
    results_dir = 'Results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    for i in range(0, len(min_internal_measures)):
        current_index = min_internal_measures[i]
        if not os.path.exists(results_dir +'/' + current_index):
            os.makedirs(results_dir +'/' + current_index)
        measure= np.array([m[current_index] for m in measures])
        saveResultsIndex(dataset_name,current_index+"_full", measure, results_dir)
        rank =rankdata(np.array([m[current_index] for m in measures]))
        rank = np.append(rank,dataset_name)
        save_path = os.path.join(os.getcwd(), results_dir,current_index,dataset_name.split('/')[-1])
        df = pd.DataFrame(rank).transpose()
        df.to_csv(save_path +".csv",index=False, header=header) #Saving the rankings

    max_internal_measure = ["SIL", "CH","DU", "BP","MC"]
    for i in range(0, len(max_internal_measure)):
        #Max index -> the values must be multiplied by -1 to achieve the correct order of the ratings
        current_index = max_internal_measure[i]
        if not os.path.exists(results_dir + '/' + current_index):
            os.makedirs(results_dir + '/' + current_index)
        measure = np.array([m[current_index] for m in measures])
        saveResultsIndex(dataset_name, current_index + "_full", measure, results_dir)
        rank = rankdata(-1* np.array([m[current_index] for m in measures]))
        rank = np.append(rank, dataset_name)
        save_path = os.path.join(os.getcwd(), results_dir, current_index, dataset_name.split('/')[-1])
        df = pd.DataFrame(rank).transpose()
        df.to_csv(save_path + ".csv", index=False, header=header) #Saving the rankings



def get_average_ranking(measures, dataset_name, write):
    ranks = np.zeros(len(measures))
    ranks += rankdata(np.array([m['DB'] for m in measures]))
    ranks += rankdata(-1 * np.array([m['SIL'] for m in measures]))
    ranks += rankdata(-1 * np.array([m['CH'] for m in measures]))
    ranks += rankdata(-1 * np.array([m['DU'] for m in measures]))
    ranks += rankdata(-1 * np.array([m['BP'] for m in measures]))
    ranks += rankdata(-1 * np.array([m['MC'] for m in measures]))
    ranks += rankdata(np.array([m['HL'] for m in measures]))
    ranks += rankdata(np.array([m['HKK'] for m in measures]))
    ranks += rankdata(np.array([m['Xie'] for m in measures]))
    ranks += rankdata(np.array([m['Scat'] for m in measures]))
    ranks /= 10
    ranks = rankdata(ranks)
    if write == True:
        header = ['dbscan', 'mst', 'SL', 'eac', 'khm', 'km', 'psc', 'al','cl','mbk','wl','kkm','fc','dataset']
        results_dir = 'Results'
        measure="average"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if not os.path.exists(results_dir +"/" +measure):
            os.makedirs(results_dir +"/" + measure)
        rank = np.append(ranks, dataset_name)
        save_path = os.path.join(os.getcwd(), results_dir, measure, dataset_name.split('/')[-1])
        df = pd.DataFrame(rank).transpose()
        df.to_csv(save_path + ".csv", index=False, header=header)
    return ranks

#Save the labels produced by the algorithms
def saveLabels(dataset_name, alg_name, labels):
    dir='Results_Labels'
    os.mkdir(dir)
    save_path = os.path.join(os.getcwd(), dir, alg_name, dataset_name.split('/')[-1])
    pickle.dump(labels, open(save_path + ".clusters", 'wb'))


def write_measures(filename, clustering_measure):
    dataframe = pd.read_csv(filename)
    dataframe = preprocess(dataframe)
    f_name = filename.split('\\')[-1].split('.')[0]
    print(f_name)
    measures, labels = [], []
    # Algorithms without K
    #########WITHOUT-K##################################
    algorithms_without_k = ["dbscan", "mst","SL", "eac"]
    for i in range(0, len(algorithms_without_k)):
        alg = algorithms_without_k[i]
        measure, label = run(f_name, dataframe, alg)
        measures.append(measure)
        labels.append(label)
        # saveLabels(f_name, alg,label)
        print(f_name+"finish " +alg)
    #####################################################

    #Computation of the parmeter K
    rankings = get_average_ranking(measures, f_name, False)
    best = np.argmin(rankings)
    K = len(np.unique(labels[best]))
    # Algorithms with K
    #########WITH-K##################################
    algorithms_with_k = ["khm","km", "psc", "al", "cl", "mbk", "wl", "kkm", "fc"]

    for i in range(0, len(algorithms_with_k)):
        alg = algorithms_with_k[i]
        measure, label = run(f_name, dataframe, alg , K)
        measures.append(measure)
        labels.append(label)
        # saveLabels(f_name, alg, label)
        print(f_name+"finish " +alg)
    ####################################################
    if clustering_measure == "average":
        final_ranking = get_average_ranking(measures, f_name, True) # combines the values of the internal indices by average ranking.
    else:
    # For getting the results of each index:
        get_rankings_per_index_final(measures, f_name)

    return measure


def get_dataframes(datasets_folder):
    PATH = os.path.join(os.getcwd(), datasets_folder)
    EXT = '*csv'
    all_csv_files = [file
                     for path, subdir, files in os.walk(PATH)
                     for file in glob(os.path.join(path, EXT))]
    return all_csv_files

def Clustering_algorithm_evaluation(datasets_folder, clustering_measure):
    dataframes = get_dataframes(datasets_folder)
    for file in dataframes:
        write_measures(file, clustering_measure)


