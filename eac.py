import numpy as np
from sklearn.metrics.cluster import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

def init_population(ind, N, K):
    return np.random.randint(0, K, (ind,N))
    
def get_fitness(data, population):
    return np.array([silhouette_score(data, x) for x in population])

def local_kmeans(data: pd.DataFrame, population, iter_max):
    new_pop = np.zeros(population.shape)
    for ind in range(len(population)):
        indexes = [np.where(population[ind] == unq) for unq in np.unique(population[ind])]
        centroids = np.array([np.average(data.iloc[x], axis=0) for x in indexes])

        kmeans = KMeans(n_clusters=len(centroids), init=centroids, max_iter=iter_max)
        kmeans.fit(data)
        new_pop[ind] = kmeans.labels_
    return new_pop

def roulette_selection(population, fitness, n_select):

    total = sum(fitness)
    acc_fit = 0
    roulette = [-1]
    pos = 0
    new_population = []
    for fit in fitness:
        acc_fit += fit
        roulette.append(acc_fit/total)
    roulette.append(1.1)

    for i in range(n_select):
        p = np.random.rand()
        for j, value in enumerate(roulette):
            if p < value:
                pos = j
                break
        new_population.append(population[pos - 1])
    
    return np.array(new_population)

def opDivision(data: pd.DataFrame, individual):
    new_ind = individual
    labels = np.unique(individual)
    k = len(labels)
    k = np.random.randint(0, k)

    cluster = data.iloc[np.where(individual == k)]
    indexes = cluster.index.values

    if cluster.shape[0] <= 3:
        return individual
    
    centroids = np.array([np.average(cluster.values, axis=0)])

    dist = euclidean_distances(centroids, cluster.values)[0]
    centroids = np.append(centroids, cluster.values[np.argmax(dist)].reshape(1, -1), axis=0)

    d = euclidean_distances(centroids, cluster.values)
    labelA = max(labels) + 1
    labelB = labelA + 1

    for index in range(len(cluster.values)):
        if d[0][index] < d[1][index]:
            new_ind[indexes[index]] = labelA
        else:
            new_ind[indexes[index]] = labelB
    
    return new_ind

def opExclusion(data: pd.DataFrame, individual):
    labels = np.unique(individual)

    if len(labels) <= 2:
        return individual
    
    centroids = [np.average(data.iloc[np.where(individual == unq)], axis=0) for unq in labels]
    k = len(labels)
    k = np.random.randint(0, k)

    centroids.pop(k)

    dist = euclidean_distances(centroids, data)
    return np.argmin(dist.T, axis=1)

def get_EAC(data: pd.DataFrame):
    N = data.shape[0]

    n_individuals = 4
    n_cluster = 5
    itKmeans = 2
    itEAC = 1000
    window_size = 50
    window = np.zeros(window_size)
    wi = 0
    population = init_population(n_individuals, N, n_cluster)

    for it in range(itEAC):
        population = local_kmeans(data, population, itKmeans)
        fitness = get_fitness(data, population)
        best_idx = np.argmax(fitness)
        best_fitness = fitness[best_idx]
        best_ind = population[best_idx]

        population = roulette_selection(population, fitness, n_individuals)

        idx = np.arange(n_individuals)
        np.random.shuffle(idx)

        setA = int(n_individuals / 2)

        for i in range(setA):
            population[idx[i]] = opDivision(data, population[idx[i]])

        for i in range(setA, n_individuals):
            population[idx[i]] = opExclusion(data, population[idx[i]])
        
        fitness = get_fitness(data, population)
        min_idx = np.argmin(fitness)
        population[min_idx] = best_ind
        fitness[min_idx] = best_fitness

        window[wi] = max(fitness)
        wi += 1
        
        if wi == window_size:
            if abs(np.average(window) - window[wi-1]) <= 0.001:
                break
            wi = 0
    return population[np.argmax(fitness)]



