import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
import os.path
from sklearn import preprocessing
from multiprocessing import Pool
import subprocess




#Pre processing according to the Clustering Meta-Learning paper
def preprocess(df):
    columns_to_remove = []
    columns_to_label_encode = []

    for col in df.columns:
        if df[col].nunique() == 1: #all the values are identical
            columns_to_remove.append(col)
            continue
        if df[col].nunique() == df.shape[0]:#the values are unique
            columns_to_remove.append(col)
            continue
        if df[col].isna().astype(int).sum() > df.shape[0] * 0.4:
            columns_to_remove.append(col)
            continue
        if df[col].dtype == 'object':
            columns_to_label_encode.append(col)

    df = df.drop(columns_to_remove, axis=1)
    df = df.dropna()

    for col in columns_to_label_encode:
        le = preprocessing.LabelEncoder()
        df[col] = le.fit_transform(df[col]).astype(int)
    val = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(val)
    df = pd.DataFrame(x_scaled, columns=df.columns)
    return df



#Apply PCA 
def pca_calculation(dataframe):
    pca = PCA(n_components=0.9)
    principalComponents = pca.fit_transform(dataframe)
    return principalComponents

#Graph generation
def buildGraph(cosin_similarities, data_set_name, EdgeLists_folder):
    G = nx.from_numpy_array(cosin_similarities)
    filter = [(u, v, d) for (u, v, d) in G.edges(data=True) if (u < v and d['weight'] > 0.95 and u != v)]
    G_filter = nx.Graph(filter)
    PATH = os.path.join(EdgeLists_folder, data_set_name)
    nx.write_weighted_edgelist(G_filter,PATH+".file")
    return G_filter

#PCA compuation -> COSINE similarity calculation -> Graph generation -> Nodes' embedding generation
def computation(path,EdgeLists_folder, NodesEmbedding_folder):
    dataframe = pd.read_csv(path)
    f_name = path.split('\\')[-1].split('.')[0]
    dataframe = preprocess(dataframe)
    principalComponents = pca_calculation(dataframe)
    cos = cosine_similarity(principalComponents)
    G = buildGraph(cos, f_name, EdgeLists_folder)
    print(f_name)
    path_in= os.path.join(EdgeLists_folder, f_name)
    path_out = os.path.join(NodesEmbedding_folder, f_name)
    subprocess.call("deepwalk --input " +path_in +".file --format weighted_edgelist --output " + path_out +".file")


def get_dataframes(datasets_folder):
    PATH = os.path.join(os.getcwd(), datasets_folder)
    EXT = '*csv'
    all_csv_files = [file
                     for path, subdir, files in os.walk(PATH)
                     for file in glob(os.path.join(path, EXT))]
    return all_csv_files

def Graph_Representation(datasets_folder, EdgeLists_folder, NodesEmbedding_folder):
    if not os.path.exists(EdgeLists_folder):
        os.mkdir(EdgeLists_folder)
    if not os.path.exists(NodesEmbedding_folder):
        os.mkdir(NodesEmbedding_folder)
    dataframes = get_dataframes(datasets_folder)
    for dataset_path in dataframes:
        computation(dataset_path, EdgeLists_folder, NodesEmbedding_folder)



