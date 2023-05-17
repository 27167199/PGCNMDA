import torch as th
from torch import nn
import torch.nn.functional as F
import torch
class PathGCNLayer(nn.Module):
    def __init__(self, hidden_dim, num_path, path_length1, path_length2):
        super(PathGCNLayer, self).__init__()
        self._alpha = 0.1

        self.path_weight1 = nn.Parameter(torch.Tensor(1, path_length1, hidden_dim))

        self.path_weight2 = nn.Parameter(torch.Tensor(1, path_length2, hidden_dim))
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)

        nn.init.xavier_normal_(self.path_weight1, gain=1.414)
        nn.init.xavier_normal_(self.path_weight2, gain=1.414)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, feats, paths, init_feats,flag):
        """
            feats: (num_nodes, d),
            paths: (num_path, num_nodes, path_length)
        """

        num_path = len(paths)
        results = []
        if flag==1:
            for i in range(num_path):
                path_feats = feats[paths[i]]  # (num_nodes, path_length, d)
                path_feats = (path_feats * self.path_weight1).sum(dim=1)  # (num_nodes, d)
                results.append(path_feats)
            results = sum(results) / len(results)
        else:
            for i in range(num_path):
                path_feats = feats[paths[i]]  # (num_nodes, path_length, d)
                path_feats = (path_feats * self.path_weight2).sum(dim=1)  # (num_nodes, d)
                results.append(path_feats)
            results = sum(results) / len(results)

        results = self.fc(results)
        results = F.relu(results)
        return results

class PGCN(nn.Module):
    def __init__(self, args):
        super(PGCN, self).__init__()
        self.args = args
        self.lin_m = nn.Linear(args.miRNA_number, args.in_feats, bias=False)
        self.lin_d = nn.Linear(args.disease_number, args.in_feats, bias=False)
        self.in_act = nn.ReLU()
        self.layers = nn.ModuleList([
            PathGCNLayer(args.in_feats, args.num_paths, args.path_length1, args.path_length2)
            for _ in range(args.num_layers)
        ])
       
        self.mlp = nn.Sequential()
        in_feat = 4 * args.out_feats
        for idx, out_feat in enumerate(args.mlp):
            self.mlp.add_module(str(idx), nn.Linear(in_feat, out_feat))
            in_feat = out_feat
        self._alpha = 0.1
        self.mlp.add_module('sigmoid', nn.Sigmoid())

    def forward(self, paths_mm, paths_dd, paths_md, miRNA, disease, samples):

        feats_mm = self.lin_m(miRNA)
        for layer in self.layers:
            feats_mm = layer(feats_mm, paths_mm, self.lin_m(miRNA),1)
            feats_mm = self._alpha * self.lin_m(miRNA)+ (1 - self._alpha) * feats_mm

        feats_dd = self.lin_d(disease)
        for layer in self.layers:
            feats_dd = layer(feats_dd, paths_dd,self.lin_d(disease),1)
            feats_dd = self._alpha * self.lin_d(disease) + (1 - self._alpha) * feats_dd

        feats_md = th.cat((self.lin_m(miRNA), self.lin_d(disease)), dim=0)
        for layer in self.layers:
            feats_md = layer(feats_md, paths_md, th.cat((self.lin_m(miRNA), self.lin_d(disease)), dim=0),2)
            feats_md = self._alpha * th.cat((self.lin_m(miRNA), self.lin_d(disease)), dim=0)+ (1 - self._alpha) * feats_md

        emb_mm_ass = feats_md[:self.args.miRNA_number, :]
        emb_dd_ass = feats_md[self.args.miRNA_number:, :]
        emb_mm = th.cat((feats_mm, emb_mm_ass), dim=1)
        emb_dd = th.cat((feats_dd, emb_dd_ass), dim=1)
        emb = th.cat((emb_mm[samples[:, 0]], emb_dd[samples[:, 1]]), dim=1)

        return self.mlp(emb)
