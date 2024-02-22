import argparse
import numpy as np
import torch
import os
import scipy.io as sio
import pickle
from torch.utils.data import DataLoader
from io_utils import EmotionDataset
from load_data import load_srt_raw_newPre
from train_utils import train_earlyStopping

import torch.nn as nn
import torch.nn.functional as F
from model import stratified_layerNorm
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune the pretrained model for EEG emotion recognition')
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    parser.add_argument('--timeLen', default=5, type=int,
                        help='time length in seconds')
    parser.add_argument('--learning-rate', default=0.0007, type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight-decay', default=0.015, type=float,
                        metavar='W', help='weight decay (default: 0.05)',
                        dest='weight_decay')
    parser.add_argument('--randSeed', default=7, type=int,
                        help='random seed')
    parser.add_argument('--timeFilterLen', default=60, type=int,
                        help='time filter length')
    parser.add_argument('--n_spatialFilters', default=16, type=int,
                        help='time filter length')
    parser.add_argument('--n_timeFilters', default=16, type=int,
                        help='time filter length')
    parser.add_argument('--multiFact', default=2, type=int,
                        help='time filter length')
    parser.add_argument('--normTrain', default='yes', type=str,
                        help='whether normTrain')
    parser.add_argument('--use-data', default='pretrained', type=str,
                        help='use which pretrained model')
    parser.add_argument('--cls', default=9, type=int,
                        help='how many cls to use')
    parser.add_argument('--dataset', default='both', type=str,
                        help='first or second')

    args = parser.parse_args()

    random.seed(args.randSeed)
    np.random.seed(args.randSeed)
    torch.manual_seed(args.randSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_num_threads(8)
    args.device = torch.device('cuda')

    label_type = args.cls

    if label_type == 9:
        label_type = 'cls9'
    elif label_type == 5:
        label_type = 'cls5'
    elif label_type == 2:
        label_type = 'cls2'
    elif label_type == 3:
        label_type = 'cls3'

    dataset = args.dataset

    if dataset == 'first':
        n_subs = 61
    elif dataset == 'second':
        n_subs = 62
    elif dataset == 'both':
        n_subs = 123

    timeLen = 1
    timeStep = 1
    fs = 250
    channel_norm = False
    time_norm = False
    data_len = fs * timeLen

    n_spatialFilters = args.n_spatialFilters
    n_timeFilters = args.n_timeFilters
    timeFilterLen = 60
    n_channs = 30
    multiFact = 2

    randomInit = False
    stratified = []

    data_dir = r'E:\zzx\Clisa\Validation\Classification_validation\MSHCL_analysis\runs_srt/'

    if args.use_data == 'pretrained':

        if label_type == 'cls2':
            save_dir = data_dir + r'raw_24video_batch24_dataset_%s_timeLen5_tf16_sf16_multiFact2_lr0.000700_wd0.015000_epochs80_randSeed%d_fold10_%s' % (
                args.dataset, args.randSeed, label_type)

        else:
            save_dir = data_dir + r'raw_28video_batch28_dataset_%s_timeLen%d_tf16_sf16_multiFact2_lr0.000700_wd0.015000_epochs80_randSeed7_fold10_%s_c0.100000_l1.000000' % (
                args.dataset, args.timeLen, label_type)

    print(save_dir)


    class ConvNet_baseNonlinearHead(nn.Module):
        def __init__(self, n_spatialFilters, n_timeFilters, timeFilterLen, n_channs, stratified, multiFact):
            super(ConvNet_baseNonlinearHead, self).__init__()
            self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
            self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen - 1) // 2))
            self.avgpool = nn.AvgPool2d((1, 30))
            self.spatialConv2 = nn.Conv2d(n_timeFilters, n_timeFilters * multiFact, (n_spatialFilters, 1),
                                          groups=n_timeFilters)
            self.timeConv2 = nn.Conv2d(n_timeFilters * multiFact, n_timeFilters * multiFact * multiFact, (1, 6),
                                       groups=n_timeFilters * multiFact)
            self.n_spatialFilters = n_spatialFilters
            self.n_timeFilters = n_timeFilters
            self.stratified = stratified

        def forward(self, input):
            # print(input.shape)
            if 'initial' in self.stratified:
                input = stratified_layerNorm(input, input.shape[0])

            out = self.spatialConv(input)
            out = out.permute(0, 2, 1, 3)
            out = self.timeConv(out)
            out1 = out.clone()
            out = F.elu(out)
            out = self.avgpool(out)

            if 'middle1' in self.stratified:
                out = stratified_layerNorm(out, out.shape[0])

            out = F.elu(self.spatialConv2(out))
            out = F.elu(self.timeConv2(out))
            # print(out.shape)

            if 'middle2' in self.stratified:
                out = stratified_layerNorm(out, out.shape[0])
            return out, out1


    data, label_repeat, n_samples, n_segs = load_srt_raw_newPre(timeLen, timeStep, fs, channel_norm, time_norm,
                                                                label_type)
    torch.cuda.set_device(args.gpu_index)

    n_total = int(np.sum(n_samples))
    print('n_total', n_total)
    print('n_samples', n_samples)
    print('n_segs', n_segs)

    bn = 1

    if dataset == 'first':
        n_subs = 61
    elif dataset == 'second':
        n_subs = 62
    elif dataset == 'both':
        n_subs = 123

    n_folds = 10

    # n_per = 8
    n_per = round(n_subs / n_folds)
    for fold in range(n_folds):
    # for fold in [0,1,8,9]:
        features1_de = np.zeros((n_subs, n_total, n_timeFilters, n_spatialFilters))
        print('fold', fold)

        model = ConvNet_baseNonlinearHead(n_spatialFilters, n_timeFilters, timeFilterLen,
                                          n_channs, stratified, multiFact).to(args.device)
        print(model)
        para_num = sum([p.data.nelement() for p in model.parameters()])
        print('Total number of parameters:', para_num)

        if not randomInit:
            if args.dataset in ['first', 'second']:
                with open(os.path.join(save_dir,
                                       'folds' + '_dataset_' + args.dataset + '_results_pretrain.pkl'),
                          'rb') as f:
                    results_pretrain = pickle.load(f)
                best_pretrain_epoch = int(results_pretrain['best_epoch'][fold])
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(best_pretrain_epoch)
                print('load:', checkpoint_name)
                # print(save_dir)
                checkpoint = torch.load(os.path.join(save_dir, str(fold), checkpoint_name), map_location=args.device)

            elif args.dataset in ['both']:
                with open(os.path.join(save_dir, 'folds_' + 'all_dataset_both_results_pretrain.pkl'), 'rb') as f:
                    results_pretrain = pickle.load(f)
                # results_pretrain['best_epoch'][0] = 47
                # results_pretrain['best_epoch'][1] = 25
                # results_pretrain['best_epoch'][2] = 19
                # results_pretrain['best_epoch'][3] = 48
                # results_pretrain['best_epoch'][4] = 25
                # results_pretrain['best_epoch'][5] = 18
                # results_pretrain['best_epoch'][6] = 75
                # results_pretrain['best_epoch'][7] = 36
                # results_pretrain['best_epoch'][8] = 48
                # results_pretrain['best_epoch'][9] = 10
                # with open(os.path.join(save_dir, 'folds_' + 'all_dataset_both_results_pretrain.pkl'), 'wb') as f:
                #     pickle.dump(results_pretrain, f)
                best_pretrain_epoch = int(results_pretrain['best_epoch'][fold])
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(best_pretrain_epoch)
                print('load:', checkpoint_name)
                checkpoint = torch.load(os.path.join(save_dir, str(fold), checkpoint_name), map_location=args.device)
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict, strict=False)

        if fold < n_folds - 1:
            val_sub = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            # val_sub = np.arange(n_per*fold, n_per*(fold+1)-1)
            val_sub = np.arange(n_per * fold, n_subs)

        val_sub = [int(val) for val in val_sub]
        print('val', val_sub)
        train_sub = list(set(np.arange(n_subs)) - set(val_sub))

        if args.normTrain == 'yes':
            print('normTrain')
            data_mean = np.mean(np.mean(data[train_sub, :, :], axis=1), axis=0)
            data_var = np.mean(np.var(data[train_sub, :, :], axis=1), axis=0)

            for i in range(n_subs):
                data[i, :, :] = (data[i, :, :] - data_mean) / np.sqrt(data_var + 1e-5)
        else:
            print('Do no norm')

        val_sub = np.arange(n_subs)

        features1_de = np.zeros((len(val_sub), n_total, n_timeFilters, n_spatialFilters))
        n = 0
        for sub in val_sub:
            data_val = data[sub, :, :]
            label_val = np.array(label_repeat)
            print(sub)
            # print('data label', data_val.shape, label_val.shape)
            # Prepare data
            valset = EmotionDataset(data_val, label_val, timeLen, timeStep, n_segs, fs)
            val_loader = DataLoader(dataset=valset, batch_size=bn, pin_memory=True, num_workers=8, shuffle=False)

            isFirst = True
            for counter, (x_batch, y_batch) in enumerate(val_loader):
                x_batch = x_batch.to(args.device)
                y_batch = y_batch.to(args.device)

                _, out = model(x_batch)
                isFirst = False
                out = out.detach().cpu().numpy()

                de = 0.5 * np.log(2 * np.pi * np.exp(1) * (np.var(out, 3)))

                if (counter + 1) * bn < n_total:
                    features1_de[n, counter * bn: (counter + 1) * bn, :, :] = de
                else:
                    features1_de[n, counter * bn:, :, :] = de
            n = n + 1

        features1_de = features1_de.reshape(len(val_sub), n_total, 256)

        de = {'de': features1_de}
        if args.normTrain == 'yes':
            sio.savemat(os.path.join(save_dir, str(fold), 'features1_de_1s_normTrain.mat'), de)
            print(os.path.join(save_dir, str(fold), 'features1_de_1s_normTrain.mat'))
        else:
            sio.savemat(os.path.join(save_dir, str(fold), 'features1_de_1s.mat'), de)
            print(os.path.join(save_dir, str(fold), 'features1_de_1s.mat'))


