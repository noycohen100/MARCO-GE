import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

def initialize(ni, num_p, K):
    C = np.random.rand(K, ni)
    G = np.random.rand(num_p, ni)
    V = np.random.rand(K, ni)
    P = np.random.rand(K, num_p, ni)

    return C, G, V, P

def get_psc(data: pd.DataFrame, K):
    max_iter = 400
    vmax = 1
    w = 0.1

    num_p = data.shape[0]
    ni = data.shape[1]
    dSSE, SSEat = 1, 0
    iC = np.zeros(num_p)
    cte = 0.1
    iteration = 0
    C, G, V, P = initialize(ni, num_p, K)
    phi1, phi2, phi3, phi4 = 0,0,0,0

    while iteration < max_iter and dSSE > 0.0001:
        niC = np.zeros(K)

        for i in range(num_p):
            D = euclidean_distances(data.values[i].reshape(1, -1), C)[0]
            iDC = np.argmin(D)
            minDC = D[iDC]
            niC[iDC] = niC[iDC] + 1

            B = P[iDC][i]
            if minDC <  euclidean_distances(B.reshape(1, -1), data.values[i].reshape(1, -1))[0]:
                P[iDC][i] = C[iDC]
            
            if minDC < euclidean_distances(G[i].reshape(1, -1), data.values[i].reshape(1, -1))[0]:
                G[i] = C[iDC]
            
            phi1 = cte * np.random.rand()
            phi2 = cte * np.random.rand()
            phi3 = cte * np.random.rand()

            for j in range(len(V[iDC])):
                V[iDC][j] = ( w * V[iDC][j] ) + ( phi1 * (P[iDC][i][j]-C[iDC][j]) ) + ( phi2 * (G[i][j]-C[iDC][j]) ) + ( phi3 * (data.values[i][j] - C[iDC][j]))

            for j in range(len(V[iDC])):
                if V[iDC][j] > vmax:
                    V[iDC][j] = vmax
            
            for j in range(len(C[iDC])):
                C[iDC][j] = C[iDC][j] + V[iDC][j]

            iC[i] = iDC
        
        mwin = np.argmax(niC)
        idx = np.where(niC == 0)[0]
        phi4 = cte * np.random.rand()

        for j in range(len(idx)):
            for l in range(len(V[idx[j]])):
                V[idx[j]][l] = ( w * V[idx[j]][l] ) + ( phi4 * (C[mwin][l]-C[idx[j]][l]) )

                if V[idx[j]][l] > vmax:
                    V[idx[j]][l] = vmax

                C[idx[j]][l] = C[idx[j]][l] + V[idx[j]][l]
        
        w *= 0.98
        cte *= 0.98
        E = np.zeros(K)

        indexes = [np.where(iC == unq)[0] for unq in np.unique(iC)]

        for i in range(len(indexes)):
            idx = indexes[i]
            for j in range(len(idx)):
                E[i] += pow(euclidean_distances(C[i].reshape(1, -1), data.values[idx[j]].reshape(1, -1))[0], 2)
        
        SSEan = SSEat
        SSEat = sum(E)
        dSSE = abs(SSEan - SSEat)
        iteration += 1
    
    return iC



