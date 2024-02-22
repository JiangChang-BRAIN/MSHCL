import os
import scipy.io as sio
import numpy as np
from reorder_vids import video_order_load, reorder_vids, reorder_vids_back
import random
import argparse


parser = argparse.ArgumentParser(description='Finetune the pretrained model for EEG emotion recognition')
parser.add_argument('--timeLen', default=5, type=int,
                    help='time length in seconds')
parser.add_argument('--use-data', default='pretrained', type=str,
                    help='what data to use')
parser.add_argument('--normTrain', default='yes', type=str,
                    help='whether normTrain')
parser.add_argument('--n-vids', default=28, type=int,
                    help='use how many videos')
parser.add_argument('--randSeed', default=7, type=int,
                    help='random seed')
parser.add_argument('--dataset', default='both', type=str,
                    help='first or second')
parser.add_argument('--cls', default=9, type=int,
                    help='how many cls to use')

args = parser.parse_args()

random.seed(args.randSeed)
np.random.seed(args.randSeed)

use_features = args.use_data
normTrain = args.normTrain

label_type = args.cls

if label_type == 9:
    label_type = 'cls9';
    n_vids = 28
elif label_type == 2:
    label_type = 'cls2'
    n_vids = 24
elif label_type == 3:
    label_type = 'cls3'
    n_vids = 28

if use_features == 'pretrained':
    if label_type == 'cls2':
        save_dir = 'runs_srt/raw_24video_batch24_dataset_%s_timeLen5_tf16_sf16_multiFact2_lr0.000700_wd0.015000_epochs80_randSeed%d_fold10_%s' % (
        args.dataset, args.randSeed, label_type)
    elif label_type == 'cls9':
        save_dir = 'runs_srt/raw_28video_batch28_dataset_%s_timeLen5_tf16_sf16_multiFact2_lr0.000700_wd0.015000_epochs80_randSeed%d_fold10_%s_c0.100000_l1.000000' % (
        args.dataset, args.randSeed, label_type)

bn_val = 1
# rn_momentum = 0.995
# print(rn_momentum)
# momentum = 0.9

n_total = 30 * n_vids
n_counters = int(np.ceil(n_total / bn_val))

dataset = args.dataset
if dataset == 'first':
    n_subs = 61
elif dataset == 'second':
    n_subs = 62
elif dataset == 'both':
    n_subs = 123

n_folds = 10
n_per = round(n_subs / n_folds)

vid_order = video_order_load(args.dataset, 28)

for decay_rate in [0.990]:
    print(decay_rate)
    for fold in range(n_folds):
        print(fold)

        if (use_features == 'pretrained'):
            if normTrain == 'yes':
                data = sio.loadmat(os.path.join(save_dir, str(fold), 'features1_de_1s_normTrain.mat'))['de']
            else:
                data = sio.loadmat(os.path.join(save_dir, str(fold), 'features1_de_1s.mat'))['de']
        print(data.shape)

        if fold < n_folds - 1:
            val_sub = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            # val_sub = np.arange(n_per*fold, n_per*(fold+1)-1)
            val_sub = np.arange(n_per * fold, n_subs)
        val_sub = [int(val) for val in val_sub]
        print('val:', val_sub)
        train_sub = list(set(np.arange(n_subs)) - set(val_sub))

        # vid order just need to read one time

        data, vid_play_order_new = reorder_vids(data, vid_order)
        print('Shape of new order: ', vid_play_order_new.shape)

        data[np.isnan(data)] = -30
        # data[data<=-30] = -30

        data_mean = np.mean(np.mean(data[train_sub, :, :], axis=1), axis=0)
        data_var = np.mean(np.var(data[train_sub, :, :], axis=1), axis=0)

        data_norm = np.zeros_like(data)
        for sub in range(data.shape[0]):
            running_sum = np.zeros(data.shape[-1])
            running_square = np.zeros(data.shape[-1])
            decay_factor = 1.
            for counter in range(n_counters):
                data_one = data[sub, counter * bn_val: (counter + 1) * bn_val, :]
                running_sum = running_sum + data_one
                running_mean = running_sum / (counter + 1)
                # running_mean = counter / (counter+1) * running_mean + 1/(counter+1) * data_one
                running_square = running_square + data_one ** 2
                running_var = (running_square - 2 * running_mean * running_sum) / (counter + 1) + running_mean ** 2

                # print(decay_factor)
                curr_mean = decay_factor * data_mean + (1 - decay_factor) * running_mean
                curr_var = decay_factor * data_var + (1 - decay_factor) * running_var
                decay_factor = decay_factor * decay_rate

                # print(running_var[:3])
                # if counter >= 2:
                data_one = (data_one - curr_mean) / np.sqrt(curr_var + 1e-5)
                data_norm[sub, counter * bn_val: (counter + 1) * bn_val, :] = data_one

        data_norm = reorder_vids_back(data_norm, vid_play_order_new)
        de = {'de': data_norm}

        if (use_features == 'pretrained'):
            if normTrain == 'yes':
                save_file = os.path.join(save_dir, str(fold),
                                         'features1_de_1s_normTrain_rnPreWeighted%.3f.mat' % decay_rate)
            else:
                save_file = os.path.join(save_dir, str(fold),
                                         'features1_de_1s_rnPreWeighted%.3f_play_order.mat' % decay_rate)
            # 获取 save_file 的目录部分
            dir_name = os.path.dirname(save_file)

            # 检查目录是否存在，如果不存在，则创建
            print(save_file)
            sio.savemat(save_file, de)


