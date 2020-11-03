import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def _calculate_perf(D, num_rows, K, p):
    val=0
    out = 0
    for i in range(num_rows):
        val = 0
        for j in range(K):
            val = val + (1 / pow(D[i][j],p))
        out = out + (K / val)
    return out

def get_khmeans(dataframe: pd.DataFrame, K: int, max_iter:int):
    min_value = pow(10, -6)
    p = 2.5
    iteration = 0
    N = dataframe.shape[0]
    M = dataframe.shape[1]
    # choosing the centers
    centers=[]
    
    while len(centers) < K:
        center = np.random.randint(10000**2) % N
        if center not in centers:
            centers.append(center)
    
    centroids = [dataframe.values[index] for index in centers]
    new_perf=-1
    old_perf = -1
    while iteration < max_iter:
        D = cdist(dataframe.values, centroids)
        for i in range(N):
            for j in range(len(centroids)):
                if D[i][j] == 0:
                    D[i][j] = min_value 
        
        
        new_perf = _calculate_perf(D, N, K, p)
        if new_perf == old_perf:
            break
        old_perf = new_perf

        alpha = np.zeros(N)

        for i in range(N):
            val = 0
            for j in range(K):
                val += (1/pow(D[i][j], p))
            alpha[i] = 1 / pow(val, 2)
        
        qnk = np.zeros((N, K))
        for i in range(N):
            for j in range(K):
                qnk[i][j] = alpha[i] / pow(D[i][j], p+2)
        
        qk = np.zeros(K)
        for i in range(K):
            for j in range(N):
                qk[i] = qk[i] + qnk[j][i]
        
        pnk = np.zeros((N, K))
        for i in range(N):
            for j in range(K):
                pnk[i][j] = qnk[i][j] / qk[j]
        
        for c in range(K):
            mk = np.zeros((N, M))
            for obj in range(N):
                for atr in range(M):
                    mk[obj][atr] = pnk[obj][c] * dataframe.values[obj][atr]
            
            for atr in range(M):
                centroids[c][atr] = 0
                for obj in range(N):
                    centroids[c][atr] += mk[obj][atr]
        
        iteration += 1

    labels = np.argmin(cdist(dataframe.values, centroids), axis=1)
    return labels

    
