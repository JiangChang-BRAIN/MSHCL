import argparse
import numpy as np
import torch
import os
import scipy.io as sio
import pickle
from torch.utils.data import DataLoader
from io_utils import EmotionDataset, TrainSampler, TrainSampler_video
from load_data import load_srt_raw_newPre
from model import ConvNet_baseNonlinearHead
from simCLR import SimCLR
from train_utils import train_earlyStopping

# from data_aug import temporal_masking, temporal_shift, interp_channel, add_gaussian_noise, scale_amplitude, bs_filter
import random
import geoopt

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune the pretrained model for EEG emotion recognition')
    parser.add_argument('--epochs-finetune', default=100, type=int, metavar='N',
                        help='number of total epochs to run in finetuning')
    parser.add_argument('--max-tol', default=20, type=int, metavar='N',
                        help='number of max tolerence for epochs with no val loss decrease in finetuning')
    parser.add_argument('--batch-size-finetune', default=270, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--learning-rate-finetune', default=0.0005, type=float, metavar='LR',
                        help='learning rate in finetuning')

    # parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

    parser.add_argument('--epochs-pretrain', default=80, type=int, metavar='N',
                        help='number of total epochs to runn pretraining')
    parser.add_argument('--restart_times', default=3, type=int, metavar='N',
                        help='number of total epochs to run in pretraining')
    parser.add_argument('--max-tol-pretrain', default=30, type=int, metavar='N',
                        help='number of max tolerence for epochs with no val loss decrease in pretraining')
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help=' n views in contrastive learning')
    parser.add_argument('--batch-size-pretrain', default=28, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--learning-rate', default=0.0007, type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight-decay', default=0.015, type=float,
                        metavar='W', help='weight decay (default: 0.05)',
                        dest='weight_decay')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--n-times', default=1, type=int,
                        help='number of sampling times for one sub pair (in one session)')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')

    parser.add_argument('--sample-method', default='cross', type=str,
                        help='how to sample pretrain data')
    parser.add_argument('--tuneMode', default='linear', type=str,
                        help='how to finetune the parameters')
    parser.add_argument('--hidden-dim', default=30, type=int,
                        help='number of hidden units')
    parser.add_argument('--timeLen', default=5, type=int,
                        help='time length in seconds')
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
    parser.add_argument('--cls', default=9, type=int,
                        help='classifications type')
    parser.add_argument('--dataset', default='both', type=str,
                        help='first or second or both')
    parser.add_argument('--training-fold', default='all', type=str,
                        help='the number of training fold, 0~9,and 9 for the subs leaf')

    parser.add_argument('--c', default=0.1, type=float,
                        help='curv (default: 0.07)')

    parser.add_argument('--lamda', default=0.1, type=float)

    args = parser.parse_args()

    random.seed(args.randSeed)
    np.random.seed(args.randSeed)
    torch.manual_seed(args.randSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_num_threads(8)
    sample_method = args.sample_method
    pretrain = True
    finetune = False
    randomInit = True
    fixFirstLayers = False
    tuneMode = args.tuneMode
    stratified = ['initial', 'middle1', 'middle2']
    channel_norm = False
    time_norm = False

    label_type = args.cls

    if label_type == 9:
        label_type = 'cls9'
        # args.batch_size_pretrain = 28
    elif label_type == 2:
        label_type = 'cls2'
        args.batch_size_pretrain = 24
    elif label_type == 3:
        label_type = 'cls3'
        # args.batch_size_pretrain = 28

    n_spatialFilters = args.n_spatialFilters
    n_timeFilters = args.n_timeFilters
    timeFilterLen = args.timeFilterLen
    multiFact = 2
    hidden_dim = args.hidden_dim
    out_dim = 30
    n_channs = 30
    fs = 250

    # The current method only supports timeLen and timeStep that can 整除. So timeStep would be better 1.
    timeLen = args.timeLen
    # timeStep = int(np.floor(args.timeLen / 3))
    timeStep = 2
    print('timeLen', args.timeLen, 'timeStep', timeStep)
    data_len = fs * timeLen

    for pos in stratified:
        assert pos in ['initial', 'middle1', 'middle2', 'final', 'final_batch', 'middle1_batch', 'middle2_batch', 'no']

    data, label_repeat, n_samples, n_segs = load_srt_raw_newPre(timeLen, timeStep, fs, channel_norm, time_norm,
                                                                label_type)

    args.device = torch.device('cuda')
    torch.cuda.set_device(args.gpu_index)

    dataset = args.dataset
    if dataset == 'first':
        n_subs = 61
    elif dataset == 'second':
        # Make sure you have deleted the sub000~060 first
        n_subs = 62
    elif dataset == 'both':
        n_subs = 123

    print('n_subs ', n_subs)

    n_folds = 10
    print('n_folds ', n_folds)

    # Here I only select 10 fold for k-fold
    if args.training_fold == 'all':
        folds_list = np.arange(n_folds-2)
    else:
        # training_fold = 0~9
        folds_list = [int(args.training_fold)]

    n_per = round(n_subs / n_folds)
    print('n_per ', n_per)

    if label_type == 'cls2':
        args.log_dir = r'raw_24video_batch%d_dataset_%s_timeLen%d_tf%d_sf%d_multiFact%d_lr%f_wd%f_epochs%d_randSeed%d_fold%d_%s_c%.1f_l%.1f' % (
            args.batch_size_pretrain, args.dataset, timeLen, n_timeFilters, n_spatialFilters,
            multiFact, args.learning_rate, args.weight_decay, args.epochs_pretrain, args.randSeed, n_folds, label_type, args.c, args.lamda)
    else:
        args.log_dir = r'raw_28video_batch%d_dataset_%s_timeLen%d_tf%d_sf%d_multiFact%d_lr%f_wd%f_epochs%d_randSeed%d_fold%d_%s_c%.1f_l%.1f' % (
            args.batch_size_pretrain, args.dataset, timeLen, n_timeFilters, n_spatialFilters,
            multiFact, args.learning_rate, args.weight_decay, args.epochs_pretrain, args.randSeed, n_folds, label_type, args.c, args.lamda)

    root_dir = './'

    save_dir = os.path.join(r'runs_srt', args.log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(args)

    if pretrain:
        results_pretrain = {}
        results_pretrain['train_top1_history'], results_pretrain['val_top1_history'] = np.zeros(
            (n_folds, args.epochs_pretrain)), np.zeros((n_folds, args.epochs_pretrain))
        results_pretrain['train_top5_history'], results_pretrain['val_top5_history'] = np.zeros(
            (n_folds, args.epochs_pretrain)), np.zeros((n_folds, args.epochs_pretrain))
        results_pretrain['train_loss_history'], results_pretrain['val_loss_history'] = np.zeros(
            (n_folds, args.epochs_pretrain)), np.zeros((n_folds, args.epochs_pretrain))
        results_pretrain['best_val_top1'], results_pretrain['best_val_top5'] = np.zeros(n_folds), np.zeros(n_folds)
        results_pretrain['best_val_loss'], results_pretrain['best_epoch'] = np.zeros(n_folds), np.zeros(n_folds)

        for fold in folds_list:
            print('fold', fold)
            manifold = geoopt.PoincareBall(c=args.c)
            model_pre = ConvNet_baseNonlinearHead(n_spatialFilters, n_timeFilters, timeFilterLen, n_channs,
                                                  stratified=stratified, multiFact=multiFact,
                                                  isMaxPool=False, args=args).to(args.device)
            print(model_pre)
            para_num = sum([p.data.nelement() for p in model_pre.parameters()])
            print('Total number of parameters:', para_num)

            if fold < n_folds - 1:
                val_sub = np.arange(n_per * fold, n_per * (fold + 1))
            else:
                # val_sub = np.arange(n_per*fold, n_per*(fold+1)-1)
                val_sub = np.arange(n_per * fold, n_subs)
            val_sub = [int(val) for val in val_sub]
            print('val', val_sub)

            train_sub = np.array(list(set(np.arange(n_subs)) - set(val_sub)))
            print('train', train_sub)

            data_train = data[list(train_sub), :, :].reshape(-1, data.shape[-1])
            label_train = np.tile(label_repeat, len(train_sub))
            print('label_train length', len(label_train))

            data_val = data[list(val_sub), :, :].reshape(-1, data.shape[-1])
            label_val = np.tile(label_repeat, len(val_sub))
            print('label_val length', len(label_val))

            trainset = EmotionDataset(data_train, label_train, timeLen, timeStep, n_segs, fs, transform=None)
            valset = EmotionDataset(data_val, label_val, timeLen, timeStep, n_segs, fs)

            print('n_samples', n_samples)
            if sample_method == 'cross':
                print('sample across subjects')
                train_sampler = TrainSampler(len(train_sub), n_times=args.n_times, batch_size=args.batch_size_pretrain,
                                             n_samples=n_samples)
                val_sampler = TrainSampler(len(val_sub), n_times=args.n_times, batch_size=args.batch_size_pretrain,
                                           n_samples=n_samples)
            elif sample_method == 'intra':
                print('sample across videos')
                train_sampler = TrainSampler_video(len(train_sub), n_times=args.n_times,
                                                   batch_size=args.batch_size_pretrain,
                                                   n_samples=n_samples)
                val_sampler = TrainSampler_video(len(val_sub), n_times=args.n_times,
                                                 batch_size=args.batch_size_pretrain,
                                                 n_samples=n_samples)

            train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=8)
            val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, pin_memory=True, num_workers=8)

            optimizer = torch.optim.Adam(model_pre.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs_pretrain, gamma=0.8, last_epoch=-1, verbose=False)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                             T_0=args.epochs_pretrain // args.restart_times,
                                                                             eta_min=0,
                                                                             last_epoch=-1)

            with torch.cuda.device(args.gpu_index):
                simclr = SimCLR(args=args, model=model_pre, optimizer=optimizer, scheduler=scheduler,
                                log_dir=os.path.join(save_dir, str(fold)), stratified='no', manifold=manifold)
                # model_pre, best_epoch, train_top1_history, val_top1_history, train_top5_history, val_top5_history, train_loss_history, val_loss_history = simclr.train(
                #     train_loader, val_loader, args.n_times, n_videos=len(n_samples))
                model_pre, best_epoch, train_top1_history, val_top1_history, train_top5_history, val_top5_history, train_loss_history, val_loss_history = simclr.train(
                    train_loader, val_loader, args.n_times)

            results_pretrain['train_top1_history'][fold, :], results_pretrain['val_top1_history'][fold,
                                                             :] = train_top1_history, val_top1_history
            results_pretrain['train_top5_history'][fold, :], results_pretrain['val_top5_history'][fold,
                                                             :] = train_top5_history, val_top5_history
            results_pretrain['train_loss_history'][fold, :], results_pretrain['val_loss_history'][fold,
                                                             :] = train_loss_history, val_loss_history
            results_pretrain['best_val_top1'][fold] = results_pretrain['val_top1_history'][fold, best_epoch]
            results_pretrain['best_val_top5'][fold] = results_pretrain['val_top5_history'][fold, best_epoch]
            results_pretrain['best_val_loss'][fold] = results_pretrain['val_loss_history'][fold, best_epoch]
            results_pretrain['best_epoch'][fold] = best_epoch

            np.save(os.path.join(save_dir, str(fold), 'train_top1_history.npy'), train_top1_history)
            np.save(os.path.join(save_dir, str(fold), 'val_top1_history.npy'), val_top1_history)
            np.save(os.path.join(save_dir, str(fold), 'train_top5_history.npy'), train_top5_history)
            np.save(os.path.join(save_dir, str(fold), 'val_top5_history.npy'), val_top5_history)
            np.save(os.path.join(save_dir, str(fold), 'train_loss_history.npy'), train_loss_history)
            np.save(os.path.join(save_dir, str(fold), 'val_loss_history.npy'), val_loss_history)

        with open(os.path.join(save_dir,
                               'folds_' + args.training_fold + '_dataset_' + args.dataset + '_results_pretrain.pkl'),
                  'wb') as f:
            pickle.dump(results_pretrain, f)
        print(save_dir)
        if args.training_fold == 'all':
            print(
                'val loss mean: %.3f, std: %.3f; val acc top1 mean: %.3f, std: %.3f; val acc top5 mean: %.3f, std: %.3f' % (
                    np.mean(results_pretrain['best_val_loss']), np.std(results_pretrain['best_val_loss']),
                    np.mean(results_pretrain['best_val_top1']), np.std(results_pretrain['best_val_top1']),
                    np.mean(results_pretrain['best_val_top5']), np.std(results_pretrain['best_val_top5'])))
        else:
            print(
                'val loss mean: %.3f, std: %.3f; val acc top1 mean: %.3f, std: %.3f; val acc top5 mean: %.3f, std: %.3f' % (
                    (results_pretrain['best_val_loss'][fold]), (results_pretrain['best_val_loss'][fold]),
                    (results_pretrain['best_val_top1'][fold]), (results_pretrain['best_val_top1'][fold]),
                    (results_pretrain['best_val_top5'][fold]), (results_pretrain['best_val_top5'][fold])))