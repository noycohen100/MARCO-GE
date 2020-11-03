from numpy import genfromtxt
from sklearn.preprocessing import LabelEncoder
import scipy
from scipy.stats import rankdata
import xgboost as xgb
import numpy as np
import os
import pandas as pd

def model(dir, depth, trees, l_r, col_s, sub_s, min_c, seed_v, number_of_features, training_file_name, number_of_algs, clustering_measure):
    dataset_name_col = 'dataset'
    target_col = "performance"
    alg_col = "alg"
    representations = []
    datasets_list = []
    rankings = pd.read_csv(ranking_file_name, index_col=0)
    datasets_counter = 0
    for file in sorted(os.listdir(dir), key=lambda s: s.lower()):
        meta_features = genfromtxt(dir + file, delimiter=',')
        file_name = file.split('.')[0]
        datasets_list.append(file_name)
        meta_features = np.append(file_name, meta_features)
        dataset_ranking = rankings.loc[rankings[dataset_name_col] == file_name]
        datasets_counter = datasets_counter + 1
        for i in range(0, number_of_algs):
            alg_name = dataset_ranking.columns[i]
            average_ranking = dataset_ranking[alg_name]
            meta_features_current = meta_features
            meta_features_current = np.append(meta_features_current, alg_name)
            meta_features_current = np.append(meta_features_current, average_ranking)
            representations.append(meta_features_current)
    columns = []
    columns.append(dataset_name_col)
    for i in range(0, number_of_features):
        columns.append("f" + str(i))
    columns.append(alg_col)
    columns.append(target_col)

    df = pd.DataFrame(representations, columns=columns)

    no_need_attribute = [target_col, dataset_name_col]
    for col in df.columns:
        if col != dataset_name_col and col != alg_col:
            df[col] = df[col].astype(float)
    MRR = []
    SRC = []
    for dataset_test in datasets_list:
        df_train = df.drop(df[df.dataset == dataset_test].index)
        df_test = df.loc[df[dataset_name_col] == dataset_test]
        y_train = df_train[target_col]
        x_train = df_train.drop(no_need_attribute, axis=1)
        nonnumeric_columns = [alg_col]
        y_test = df_test[target_col]
        x_test = df_test.drop(no_need_attribute, axis=1)
        le = LabelEncoder()
        for feature in nonnumeric_columns:
            x_train[feature] = le.fit_transform(x_train[feature])
            x_test[feature] = le.fit_transform(x_test[feature])
        gbm = xgb.XGBRegressor(max_depth=depth, learning_rate=l_r, n_estimators=trees,
                               objective='rank:pairwise',
                               min_child_weight=min_c, colsample_bytree=col_s, subsample=sub_s, seed=seed_v)

        gbm.fit(x_train, y_train)
        prediction = gbm.predict(x_test)
        prediction_rank = rankdata(prediction)
        y_r = rankdata(y_test)
        p_r = rankdata(prediction)
        spr = np.abs(scipy.stats.spearmanr(p_r, y_r)[0])

        SRC.append(spr)
        index_min_test = np.argmin(y_test.values)
        mrr = 1 / prediction_rank[index_min_test]
        MRR.append(mrr)

        #########################################################################
    print("MRR = " + str(np.mean(MRR)))
    print("SRC = " + str(np.mean(SRC)))

    return SRC, MRR, datasets_list
