import argparse
from utils import get_data, data_processing, make_adj, get_gaussian
from train import train
import os
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np

def result(args):
    data = get_data(args)
    mf = data['mf']
    dss = data['dss']
    args.miRNA_number = data['miRNA_number']
    args.disease_number = data['disease_number']
    data_processing(data, args)

    samples1 = data['train_samples1']
    samples0 = data['train_samples0']

    kf = KFold(n_splits=int(args.kfolds), shuffle=True, random_state=5)
    kf_samples1 = [train_test for train_test in kf.split(samples1)]
    kf_samples0 = [train_test for train_test in kf.split(samples0)]
    auc_list = []
    aupr_list = []
    acc_list = []
    prec_list = []
    recall_list = []
    f1_list = []
    for k in range(args.kfolds):
        train_samples = np.concatenate((samples1[kf_samples1[k][0]], samples0[kf_samples0[k][0]]), axis=0)
        test_samples = np.concatenate((samples1[kf_samples1[k][1]], samples0[kf_samples0[k][1]]), axis=0)
        train_adj = make_adj(samples1[kf_samples1[k][0], :-1], ((mf.shape[0]), dss.shape[0]))
        GM_train = get_gaussian(train_adj)
        GD_train = get_gaussian(train_adj.T)
        M_train = mf * np.where(mf > 0, 1, 0) + GM_train * np.where(mf > 0, 0, 1)
        D_train = dss * np.where(dss > 0, 1, 0) + GD_train * np.where(dss > 0, 0, 1)
        data['ms'] = M_train
        data['ds'] = D_train
        data['train_md'] = samples1[kf_samples1[k][0], :-1]
        test_score = train(data, args, train_samples, test_samples)
        all_output = test_score.detach().numpy().flatten()
        all_targets = test_samples[:, 2]
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_output)
        auc = metrics.auc(fpr, tpr)
        pr, re, thresholds2 = metrics.precision_recall_curve(all_targets, all_output)
        aupr = metrics.auc(re, pr)
        f1_s = [2 * p * r / (p + r) for p, r in zip(pr, re)]
        optimal_f1_index = np.argmax(f1_s)
        optimal_threshold2 = thresholds2[optimal_f1_index]
        print(optimal_threshold2)
        all_scores = []
        for p in all_output:
            if p > optimal_threshold2:
                all_scores.append(1)
            else:
                all_scores.append(0)
        all_scores = np.array(all_scores)
        accuracy = metrics.accuracy_score(all_targets, all_scores)
        precision = metrics.precision_score(all_targets, all_scores)
        recall = metrics.recall_score(all_targets, all_scores)
        f1 = metrics.f1_score(all_targets, all_scores)
        print('fold:', k+1, '\n  AUC = \t', auc,  '\n  AUPR = \t', aupr,
              '\n  Acc = \t', accuracy, '\n prec = \t', precision,
              '\n  recall = \t',  recall, '\n f1_score = \t', f1)

        auc_list.append(auc)
        aupr_list.append(aupr)
        acc_list.append(accuracy)
        prec_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    auc_arr = np.array(auc_list)
    aupr_arr = np.array(aupr_list)
    acc_arr = np.array(acc_list)
    prec_arr = np.array(prec_list)
    recall_arr = np.array(recall_list)
    f1_arr = np.array(f1_list)

    ave_auc = np.mean(auc_arr)
    ave_aupr = np.mean(aupr_arr)
    ave_acc = np.mean(acc_arr)
    ave_prec = np.mean(prec_arr)
    ave_recall = np.mean(recall_arr)
    ave_f1 = np.mean(f1_arr)

    print('ave: \n  AUC = \t', ave_auc, '\n  AUPR = \t', ave_aupr,
          '\n  Acc = \t', ave_acc, '\n prec = \t', ave_prec,
          '\n  recall = \t', ave_recall, '\n f1_score = \t', ave_f1)


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
parser.add_argument("--kfolds",type=int, default=10, help="10-cv")

args = parser.parse_args()
args.data_dir = '../data/' + args.dataset + '/'


result(args)
