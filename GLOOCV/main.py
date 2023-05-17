import argparse
from utils import get_data, data_processing, make_adj, get_gaussian
from train import train
import os
from sklearn import metrics
import numpy as np

def Get_test_train(length,data,i):
    test_data=data[i]
    train_data=data[:]
    train_data=np.delete(train_data,i,0)
    return train_data,test_data

def result(args):
    data = get_data(args)
    mf = data['mf']
    dss = data['dss']
    args.miRNA_number = data['miRNA_number']
    args.disease_number = data['disease_number']
    data_processing(data, args)

    samples1 = data['train_samples1']
    samples0 = data['train_samples0']
    unsamples =data["unsamples"]
    auc_list = []

    for i in range(len(samples1)):
        train_samples1, test_samples1=Get_test_train(len(samples1),samples1,i)
        train_samples =np.vstack((train_samples1,samples0))
        test_samples = np.vstack((test_samples1,unsamples))
        train_adj = make_adj(train_samples1[:, :-1], ((mf.shape[0]), dss.shape[0]))
        GM_train = get_gaussian(train_adj)
        GD_train = get_gaussian(train_adj.T)
        M_train = mf * np.where(mf > 0, 1, 0) + GM_train * np.where(mf > 0, 0, 1)
        D_train = dss * np.where(dss > 0, 1, 0) + GD_train * np.where(dss > 0, 0, 1)
        data['ms'] = M_train
        data['ds'] = D_train
        data['train_md'] = train_samples[:, :-1]
        test_score = train(data, args, train_samples, test_samples)
        all_output = test_score.detach().numpy().flatten()
        all_targets = test_samples[:, 2]
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_output)
        auc = metrics.auc(fpr, tpr)
        auc_list.append(auc)
        print('AUC = \t', auc)
    auc_arr = np.array(auc_list)
    ave_auc = np.mean(auc_arr)
    print('ave_AUC = \t', ave_auc)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--fm', type=int, default=64, help='length of miRNA feature')
parser.add_argument('--fd', type=int, default=64, help='length of dataset feature')
parser.add_argument('--wd', type=float, default=1e-3, help='weight_decay')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument("--in_feats", type=int, default=64, help='Input layer dimensionalities.')
parser.add_argument("--hid_feats", type=int, default=64, help='Hidden layer dimensionalities.')
parser.add_argument("--out_feats", type=int, default=64, help='Output layer dimensionalities.')
parser.add_argument("--num_layers", type=int, default=2, help='Number of GNN layers.')
parser.add_argument('--random_seed', type=int, default=123, help='random seed')
parser.add_argument('--mlp', type=list, default=[64, 1], help='mlp layers')
parser.add_argument('--neighbor', type=int, default=30, help='neighbor')
parser.add_argument('--dataset', default='HMDD v2.0', help='dataset')
parser.add_argument('--negative_rate', type=float,default=1.0, help='negative_rate')
parser.add_argument("--num_paths", type=int, default=1, )
parser.add_argument("--path_length1", type=int, default=1)
parser.add_argument("--path_length2", type=int, default=1)


args = parser.parse_args()
args.data_dir = '../data/' + args.dataset + '/'


result(args)
