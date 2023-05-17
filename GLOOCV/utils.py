import math
import random
import numpy as np
import torch as th


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return th.LongTensor(edge_index)


def make_adj(edges, size):
    edges_tensor = th.LongTensor(edges).t()
    values = th.ones(len(edges))
    adj = th.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    return adj



def data_processing(data, args):
    md_matrix = make_adj(data['md'], (args.miRNA_number, args.disease_number))
    one_index = []
    zero_index = []
    for i in range(md_matrix.shape[0]):
        for j in range(md_matrix.shape[1]):
            if md_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)
    unsamples=[]
    if args.negative_rate == -1:
        zero_index = zero_index
    else:
        unsamples = zero_index[int(args.negative_rate * (len(one_index)-1)):]
        zero_index = zero_index[:int(args.negative_rate * (len(one_index)-1))]
    index1 = np.array(one_index, np.int)
    label1 = np.array([1] * len(one_index), dtype=np.int)
    samples1 = np.concatenate((index1, np.expand_dims(label1, axis=1)), axis=1)
    index0 = np.array(zero_index, np.int)
    label0 = np.array([0] * len(zero_index), dtype=np.int)
    samples0 = np.concatenate((index0, np.expand_dims(label0, axis=1)), axis=1)

    data['unsamples']= np.concatenate((np.array(unsamples,np.int), np.expand_dims(np.array([0] * len(unsamples), dtype=np.int), axis=1)), axis=1)

    index = np.array(one_index + zero_index, np.int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=np.int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    data['train_samples'] = samples
    data['train_samples1'] = samples1
    data['train_samples0'] = samples0


def k_matrix(matrix, k=20):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)

def get_data(args):
    data = dict()
    mf = np.loadtxt(args.data_dir + 'miRNA functional similarity matrix.txt', dtype=np.float)
    dss1 = np.loadtxt(args.data_dir + 'disease functional similarity matrix.txt', dtype=np.float)
    ds1 = np.loadtxt(args.data_dir + 'disease semantic similarity matrix 1.txt', dtype=np.float)
    ds2 = np.loadtxt(args.data_dir + 'disease semantic similarity matrix 2.txt', dtype=np.float)
    dss2 = (ds1 + ds2) / 2
    dss = (dss1 + dss2) / 2


    data['miRNA_number'] = int(mf.shape[0])
    data['disease_number'] = int(dss.shape[0])
    data['mf'] = mf
    data['dss'] = dss
    data['md'] = np.loadtxt(args.data_dir + 'known disease-miRNA association number.txt', dtype=np.int) - 1
    return data


def get_gaussian(adj):
    Gaussian = np.zeros((adj.shape[0], adj.shape[0]), dtype=np.float32)
    gamaa = 1
    sumnorm = 0
    for i in range(adj.shape[0]):
        norm = np.linalg.norm(adj[i]) ** 2
        sumnorm = sumnorm + norm
    gama = gamaa / (sumnorm / adj.shape[0])
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            Gaussian[i, j] = math.exp(-gama * (np.linalg.norm(adj[i] - adj[j]) ** 2))

    return Gaussian


