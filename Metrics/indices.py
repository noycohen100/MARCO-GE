import numpy as np

import scipy as sy
from math import pow


class internalIndex:
    def __init__(self, num_k):
        self.num_k = num_k
        self.class_iter = range(0, num_k)

    def euclidean_centroid(self, data, label, label_num=False):
        if label_num == False:
            num_attr = data.shape[1]
            centroid = np.zeros([1, num_attr])
            for attr_id in range(num_attr):
                sum_attr_id = 0
                for i in range(len(data)):
                    sum_attr_id += data[i][attr_id]
                centroid[0][attr_id] = sum_attr_id / len(data)
            return centroid[0]
        else:
            count = 0
            for l in label:
                if l == label_num:
                    count += 1
            length = count
            num_attr = data.shape[1]
            centroid = np.zeros([1, num_attr])
            for attr_id in range(num_attr):
                sum_attr_id = 0
                for i in range(len(data)):
                    if label[i] == label_num:
                        sum_attr_id += data[i][attr_id]
                if length==0:
                    length =0.0000001
                    centroid[0][attr_id] = sum_attr_id / length
                else:
                    centroid[0][attr_id] = sum_attr_id / length
            return centroid[0]

    def centroid_list(self, data, label):
        c_list = []
        for index_i in self.class_iter:
            c_list.append(self.euclidean_centroid(data, label, index_i))
        return c_list

    def element_of_clustert(self, data, label, cluster_i):
        eoc = []
        for l in range(len(data)):
            if label[l] == cluster_i:
                eoc.append(data[l])
        return np.asarray(eoc)

    def distance_from_cluster(self, data, label, cluster_i, centroid_i):
        eoc = self.element_of_clustert(data, label, cluster_i)
        centroid_i = self.euclidean_centroid(data, label, centroid_i)
        distance = 0
        for i in eoc:
            distance += sy.spatial.distance.euclidean(centroid_i, i)
        return distance

    def distance_from_cluster_sqr(self, data, label, cluster_i, centroid_i):
        eoc = self.element_of_clustert(data, label, cluster_i)
        centroid_i = self.euclidean_centroid(data, label, centroid_i)
        distance = 0
        for i in eoc:
            distance += sy.spatial.distance.sqeuclidean(centroid_i, i)
        return distance

    def cluster_stdev(self, data, label, i=False):
        if i == 'all':
            result = 0
            for c in self.class_iter:
                result += self.cluster_stdev(data, label, c)
            return (np.sqrt(result)) / self.num_k
        if i != False:
            data = self.element_of_clustert(data, label, i)
        var_vec = np.var(data, 0)
        var_vec_t = np.transpose(var_vec)
        return np.sqrt(np.dot(var_vec, var_vec_t))

    def nn_exclude(self, data, label, data_i, label_i, k):
        data_i = [data_i]
        dist_mat = sy.spatial.distance.cdist(data_i, data)[0]
        dist_mat = np.argsort(dist_mat)[1:k + 1]
        count = 0
        for i in dist_mat:
            if label_i != i:
                count = count + 1
        return count

    def dbi(self, data, label):
        db = 0
        for index_i in self.class_iter:
            c_i = self.euclidean_centroid(data, label, index_i)
            max_rij = 0
            d_i_avg = self.distance_from_cluster(data, label, index_i, index_i) / len(
                self.element_of_clustert(data, label, index_i))
            for index_j in self.class_iter:
                if index_i == index_j:
                    continue
                else:
                    c_j = self.euclidean_centroid(data, label, index_j)
                    d_j_avg = self.distance_from_cluster(data, label, index_j, index_j) / len(
                        self.element_of_clustert(data, label, index_j))
                    d_i_j = sy.spatial.distance.euclidean(c_j, c_i)
                    candidate = (d_i_avg + d_j_avg) / d_i_j
                    if candidate > max_rij:
                        max_rij = candidate
            db += max_rij
        return db / (len(np.unique(label)))

    def xie_benie(self, data, label):
        total_distance = 0
        for index_i in self.class_iter:
            total_distance += self.distance_from_cluster_sqr(data, label, index_i, index_i)
        c_list = self.centroid_list(data, label)
        min_cij=1
        if self.num_k != 1:
            min_cij = sy.spatial.distance.pdist(c_list, 'sqeuclidean').min()
        xb = total_distance / (len(data) * min_cij)
        return xb

    def dunn(self, data, label):
        min_ij_candidate = float('Infinity')
        max_ck = float('-Infinity')
        for index_k in self.class_iter:
            eoc_k = self.element_of_clustert(data, label, index_k)
            if len(eoc_k) == 1:
                candidate = 0
            else:
                candidate = sy.spatial.distance.pdist(eoc_k, 'euclidean').max()
            if candidate > max_ck:
                max_ck = candidate
        for index_i in self.class_iter:
            eoc_i = self.element_of_clustert(data, label, index_i)
            for index_j in self.class_iter:
                if index_i == index_j:
                    continue
                eoc_j = self.element_of_clustert(data, label, index_j)
                min_j = float('Infinity')
                for e_c_i in eoc_i:
                    for e_c_j in eoc_j:
                        candidate = sy.spatial.distance.euclidean(e_c_i, e_c_j)
                        if candidate < min_j:
                            min_j = candidate
                        else:
                            pass
                min_j = min_j / max_ck
                if min_j < min_ij_candidate:
                    min_ij_candidate = min_j
        return min_ij_candidate

    def CH(self, data, label):
        data_centroid = self.euclidean_centroid(data, label)
        cent_distsqr = 0
        ecent_distsqr = 0
        for index_i in self.class_iter:
            n_element_i = len(self.element_of_clustert(data, label, index_i))
            ci_centroid = self.euclidean_centroid(data, label, index_i)
            sqr_dist = sy.spatial.distance.sqeuclidean(data_centroid, ci_centroid)
            cent_distsqr = cent_distsqr + sqr_dist * n_element_i
            ecent_distsqr += self.distance_from_cluster_sqr(data, label, index_i, index_i)
        return (cent_distsqr / (self.num_k - 1)) / (ecent_distsqr / (len(data) - self.num_k))

    def I(self, data, label):
        # Need compactness, max_centroid_dist, dist_to_center, NC
        compactness = 0
        p = 2
        centroid_list = []
        for index_i in self.class_iter:
            current_centroid = self.euclidean_centroid(data, label, index_i)
            centroid_list.append(current_centroid)
            eoc = self.element_of_clustert(data, label, index_i)
            for e in eoc:
                compactness = compactness + sy.spatial.distance.euclidean(e, current_centroid)
        max_centroid_dist = sy.spatial.distance.pdist(centroid_list, 'euclidean').max()
        data_centroid = self.euclidean_centroid(data, label)
        distance_to_center = 0
        for data_p in data:
            distance_to_center = distance_to_center + sy.spatial.distance.euclidean(data_p, data_centroid)
        NC = len(self.class_iter)
        return pow((max_centroid_dist * distance_to_center) / (NC * compactness), p)

    def CVNN(self, data, label):
        # Need COM and SEP
        k = 10
        COM = 0
        SEP = []
        for index_i in self.class_iter:
            temp_sep = 0
            eoc = self.element_of_clustert(data, label, index_i)
            n = len(eoc)
            if n != 1:
                COM = COM + (sum(sy.spatial.distance.pdist(eoc, 'euclidean')) * 2 / (n * (n - 1)))
            for e in eoc:
                q_index_e = self.nn_exclude(data, label, e, index_i, k)
                temp_sep = temp_sep + (q_index_e / k)
            SEP.append(temp_sep)
        SEP = max(SEP)
        return COM, SEP

    def CVNN_n(self, COM, SEP):
        index = self.num_k - 2
        return COM[index] / max(COM) + SEP[index] / max(SEP)

    def Scat(self, data, label):
        result = 0
        for i in self.class_iter:
            result += self.cluster_stdev(data, label, i)

        return (result / ((self.num_k) * self.cluster_stdev(data, label)))

    def SD_valid(self, data, label):
        scat = self.Scat(data, label)
        centroid_list = []
        for index_i in self.class_iter:
            current_centroid = self.euclidean_centroid(data, label, index_i)
            centroid_list.append(current_centroid)
        max_cent_dist = sy.spatial.distance.pdist(centroid_list, 'euclidean').max()
        min_cent_dist = sy.spatial.distance.pdist(centroid_list, 'euclidean').min()
        seperation = 0
        for c in centroid_list:
            seperation = seperation + 1 / sum(sy.spatial.distance.cdist([c], centroid_list)[0])
        dis = seperation * max_cent_dist / min_cent_dist

        return scat, dis

    def SD_valid_n(self, scat, dis):
        index = self.num_k - 2
        return max(dis) * scat[index] + dis[index]

    def SDbw(self, data, label):
        scat = self.Scat(data, label)
        dens_bw = 0

        sij = 0
        for i in self.class_iter:
            s_i = 0
            for j in self.class_iter:
                s_j = 0
                if i == j:
                    continue
                else:

                    eoc_i = self.element_of_clustert(data, label, i)
                    eoc_j = self.element_of_clustert(data, label, j)
                    centroid_i = self.euclidean_centroid(data, label, i)
                    centroid_j = self.euclidean_centroid(data, label, j)
                    stdev_i = self.cluster_stdev(eoc_i, label, False)
                    stdev_j = self.cluster_stdev(eoc_j, label, False)
                    w_eoc_i = 0
                    w_eoc_j = 0
                    for x in eoc_i:
                        if sy.spatial.distance.euclidean(x, centroid_i) < stdev_i:
                            w_eoc_i += 1
                    for x in eoc_j:
                        if sy.spatial.distance.euclidean(x, centroid_j) < stdev_j:
                            w_eoc_j += 1
                    if w_eoc_i > w_eoc_j:
                        weight = w_eoc_i
                    else:
                        weight = w_eoc_j
                    u_ij = (np.sum([self.euclidean_centroid(data, label, i), self.euclidean_centroid(data, label, j)],
                                   axis=0)) / 2
                    eoc_ij = np.append(eoc_i, eoc_j, axis=0)
                    stdev_ij = np.sqrt(pow(stdev_i, 2) + pow(stdev_j, 2)) / 2
                    for x in eoc_ij:
                        d_ij = sy.spatial.distance.euclidean(u_ij, x)
                        if d_ij < stdev_ij:
                            weighted_d_ij = d_ij / weight
                            s_j += weighted_d_ij
                s_i += s_j
            sij += s_i
        dens_bw = sij / (self.num_k * (self.num_k - 1))
        return (scat + dens_bw)