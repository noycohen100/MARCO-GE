# MARCO-GE
MARCO-GE is a novel meta learning approach for clustering algorithm recommendation. 


Usage
-----

**Usage**
    ``python main.py --Df datasets_folder --em clustering_measure_name --mm clustering_measure_name --ne number_of_epochs --es graph_embedding_size``

**Example Usage #1 - particular clustering measure**
    ``python main.py --Df Datasets --em all --mm BP --ne 20 --es 100``

**Example Usage #2 - average ranking measure**
    ``python main.py --Df Datasets --em average --mm average --ne 10 --es 300``
    
**--Df:** path to folder with datasets. The folder must contain CSV files.

**--em:** select clustering evaluation measure/s for evaluating the performance of the clustering algorithms.
The available options are: 
- average - computes the average ranking measure.
- all - computes 10 different clustering evaluation measures.

The evaluation's results are stored in the "Results" folder.

**--mm:** select a clustering evaluation measure for training the meta-model
The available options are:
- average - computes the average ranking measure
- BP = Bezdek-Pal
- DU = Dunn Index
- CH = Calinski-Harabasz
- SIL = Silhouette score
- MC = Milligan-Cooper
- DB = Davies-Bouldin
- HKK = Handl-Knowles-Kell
- HL = Hubert-Levin
- Scat = SD-Scat 
- Xie = Xie-Ben

**--ne:** number of epochs for training the GCNN model

**--es:** the graph embedding size

Requirements
-----
- python 3.6
- mst_clustering
- pandas
- networkx=2.4
- numpy=1.16.0
- xgboost
- sklearn_extensions
- openensembles
- scikit-learn=0.20.2
- dgl=0.4.3
- deepwalk
After installing deepwalk package, follow the instructions in: https://github.com/shun1024/weighted-deepwalk

Citing
-----
If you find MARCO-GE useful in your research. we ask that you cite the following paper:

Acknowledgments
-----
The implementation of several clustering algorithms and clustering measures were taken from:
https://github.com/ItayGabbay/ClusteringAlgorithmSelection
