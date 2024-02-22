import argparse
import numpy as np
import torch
import os
import scipy.io as sio
import pickle
import random

def LDS(sequence):
    # print(sequence.shape) # (30, 256)

    # sequence_new = np.zeros_like(sequence) # (30, 256)
    ave = np.mean(sequence, axis=0)  # [256,]
    u0 = ave
    X = sequence.transpose((1, 0))  # [256, 30]

    V0 = 0.01
    A = 1
    T = 0.0001
    C = 1
    sigma = 1

    [m, n] = X.shape  # (1, 30)
    P = np.zeros((m, n))  # (1, 1, 30) dia
    u = np.zeros((m, n))  # (1, 30)
    V = np.zeros((m, n))  # (1, 1, 30) dia
    K = np.zeros((m, n))  # (1, 1, 30)

    K[:, 0] = (V0 * C / (C * V0 * C + sigma)) * np.ones((m,))
    u[:, 0] = u0 + K[:, 0] * (X[:, 0] - C * u0)
    V[:, 0] = (np.ones((m,)) - K[:, 0] * C) * V0

    for i in range(1, n):
        P[:, i - 1] = A * V[:, i - 1] * A + T
        K[:, i] = P[:, i - 1] * C / (C * P[:, i - 1] * C + sigma)
        u[:, i] = A * u[:, i - 1] + K[:, i] * (X[:, i] - C * A * u[:, i - 1])
        V[:, i] = (np.ones((m,)) - K[:, i] * C) * P[:, i - 1]

    X = u

    return X.transpose((1, 0))


parser = argparse.ArgumentParser(description='Finetune the pretrained model for EEG emotion recognition')

parser.add_argument('--use-data', default='pretrained', type=str,
                    help='what data to use')
parser.add_argument('--n_spatialFilters', default=16, type=int,
                    help='time filter length')
parser.add_argument('--n_timeFilters', default=16, type=int,
                    help='time filter length')
parser.add_argument('--normTrain', default='yes', type=str,
                    help='whether normTrain')
parser.add_argument('--n-vids', default=28, type=int,
                    help='use how many videos')
parser.add_argument('--randSeed', default=7, type=int,
                    help='random seed')
parser.add_argument('--smooth-length', default=30, type=int,
                    help='the length for lds smooth')
parser.add_argument('--dataset', default='both', type=str,
                    help='first or second')
parser.add_argument('--cls', default=9, type=int,
                    help='how many cls to use')

args = parser.parse_args()


n_spatial = args.n_spatialFilters
n_time = args.n_timeFilters
random.seed(args.randSeed)
np.random.seed(args.randSeed)
# n_vids = args.n_vids

label_type = args.cls

if label_type == 9:
    label_type = 'cls9'
    n_vids = 28
elif label_type == 2:
    label_type = 'cls2'
    n_vids = 24
elif label_type == 3:
    label_type = 'cls3'
    n_vids = 28

if n_vids == 24:
    save_dir = './runs_srt/raw_24video_batch24_dataset_%s_timeLen5_tf16_sf16_multiFact2_lr0.000700_wd0.015000_epochs80_randSeed%d_fold10_cls2' % (args.dataset, args.randSeed)
elif n_vids == 28:
    save_dir = './runs_srt/raw_28video_batch28_dataset_%s_timeLen5_tf16_sf16_multiFact2_lr0.000700_wd0.015000_epochs80_randSeed7_fold10_%s_c1.0_l0.1'% (args.dataset, label_type)

dataset = args.dataset
if dataset == 'first':
    n_subs = 61
elif dataset == 'second':
    n_subs = 62
elif dataset == 'both':
    n_subs = 123

n_folds = 10
n_per = round(n_subs/n_folds)
n_length = args.smooth_length

for fold in range(n_folds):
    subs_feature_lds = np.ones((n_subs, n_vids * n_length, n_spatial * n_time))
    data_dir = os.path.join(save_dir, str(fold), 'features1_de_1s_normTrain_rnPreWeighted0.990.mat')
    feature_de_norm = sio.loadmat(data_dir)['de']
    for sub in range(n_subs):
        subs_feature_lds[sub, : ,:] = LDS(feature_de_norm[sub,:,:])
    de_lds = {'de_lds': subs_feature_lds}
    save_file = os.path.join(save_dir, str(fold), 'features1_de_1s_lds.mat')
    print(save_file)
    sio.savemat(save_file, de_lds)



