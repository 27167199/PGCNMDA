import sys
sys.path.append('../')
from model.PGCN import PGCN
from torch import optim, nn
from tqdm import trange
from utils1 import k_matrix
import dgl
import networkx as nx
import copy
import numpy as np
import torch as th
import torch

def get_random_walk_path(g, num_walks, walk_length):
    """
    Get random walk paths.
    """
    device = g.device
    g = g.to("cpu")
    walks = []
    nodes = g.nodes()

    for _ in range(num_walks):
        walks.append(
            dgl.sampling.node2vec_random_walk(g, nodes, p=1, q=1, walk_length=walk_length)
        )
    walks = torch.stack(walks).to(device) # (num_walks, num_nodes, walk_length)
    return walks

def train(data,args,train_samples, test_samples):
    model = PGCN(args)
    optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
    cross_entropy = nn.BCELoss()
    epochs = trange(args.epochs, desc='train')
    miRNA=data['ms']
    disease=data['ds']


    for _ in epochs:
        model.train()
        optimizer.zero_grad()
        mm_matrix = k_matrix(data['ms'], args.neighbor)
        dd_matrix = k_matrix(data['ds'], args.neighbor)

        mm_nx=nx.from_numpy_matrix(mm_matrix)
        dd_nx=nx.from_numpy_matrix(dd_matrix)
        mm_graph = dgl.from_networkx(mm_nx)
        dd_graph = dgl.from_networkx(dd_nx)
        md_copy = copy.deepcopy(data['train_md'])
        md_copy[:, 1] = md_copy[:, 1] + args.miRNA_number
        md_graph = dgl.graph(
            (np.concatenate((md_copy[:, 0], md_copy[:, 1])), np.concatenate((md_copy[:, 1], md_copy[:, 0]))),
            num_nodes=args.miRNA_number + args.disease_number)

        mm_graph = mm_graph.remove_self_loop().add_self_loop()
        dd_graph = dd_graph.remove_self_loop().add_self_loop()
        md_graph = md_graph.remove_self_loop().add_self_loop()
        paths_mm = get_random_walk_path(mm_graph, args.num_paths, args.path_length1-1)
        paths_dd = get_random_walk_path(dd_graph, args.num_paths, args.path_length1-1)
        paths_md = get_random_walk_path(md_graph, args.num_paths, args.path_length2-1)

        miRNA_th = th.Tensor(miRNA)
        disease_th = th.Tensor(disease)
        train_samples_th = th.Tensor(train_samples).float()
        train_score = model(paths_mm, paths_dd, paths_md, miRNA_th, disease_th, train_samples)
        train_loss = cross_entropy(th.flatten(train_score), train_samples_th[:, 2])
        train_loss.backward()
        optimizer.step()

    model.eval()
    test_score = model(paths_mm, paths_dd, paths_md, miRNA_th, disease_th, test_samples)

    return test_score