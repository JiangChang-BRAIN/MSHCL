import argparse
import numpy as np
import pandas as pd
import torch
import os
import scipy.io as sio
import pickle
from torch.utils.data import DataLoader
from io_utils import DEDataset, TrainSampler_sub
from load_data import load_srt_pretrainFeat
from model import simpleNN3
from simCLR import SimCLR
from train_utils import train_earlyStopping
import random



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune the pretrained model for EEG emotion recognition')
    parser.add_argument('--epochs-finetune', default=100, type=int, metavar='N',
                        help='number of total epochs to run in finetuning')
    parser.add_argument('--max-tol', default=50, type=int, metavar='N',
                        help='number of max tolerence for epochs with no val loss decrease in finetuning')
    parser.add_argument('--batch-size-finetune', default=270, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--learning-rate-finetune', default=0.0005, type=float, metavar='LR',
                        help='learning rate in finetuning')

    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

    parser.add_argument('--epochs-pretrain', default=100, type=int, metavar='N',
                        help='number of total epochs to run in pretraining')
    parser.add_argument('--restart_times', default=3, type=int, metavar='N',
                        help='number of total epochs to run in pretraining')
    parser.add_argument('--max-tol-pretrain', default=30, type=int, metavar='N',
                        help='number of max tolerence for epochs with no val loss decrease in pretraining')
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help=' n views in contrastive learning')
    parser.add_argument('--batch-size-pretrain', default=28, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--learning-rate', default=0.0005, type=float, metavar='LR',
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

    parser.add_argument('--use-data', default='raw', type=str,
                        help='use what kind of input data')
    parser.add_argument('--sample-method', default='cross', type=str,
                        help='how to sample pretrain data')
    parser.add_argument('--tuneMode', default='linear', type=str,
                        help='how to finetune the parameters')
    parser.add_argument('--hidden-dim', default=30, type=int,
                        help='number of hidden units')
    parser.add_argument('--timeLen-pretrain', default=5, type=int,
                        help='time length in seconds of pretraining')
    parser.add_argument('--randSeed', default=7, type=int,
                        help='random seed')

    parser.add_argument('--n-vids', default=28, type=int,
                        help='use how many videos')
    parser.add_argument('--timeFilterLen', default=60, type=int,
                        help='time filter length')
    parser.add_argument('--n_spatialFilters', default=16, type=int,
                        help='time filter length')
    parser.add_argument('--n_timeFilters', default=16, type=int,
                        help='time filter length')
    parser.add_argument('--multiFact', default=2, type=int,
                        help='time filter length')
    parser.add_argument('--dataset', default='both', type=str,
                        help='first or second')
    parser.add_argument('--val-method', default='10_folds', type=str,
                        help='10_folds or loo')
    parser.add_argument('--cls', default=9, type=int,
                        help='how many cls to use')
    parser.add_argument('--train-or-test', default='train', type=str, help='Using for strategy')

    args = parser.parse_args()

    random.seed(args.randSeed)
    np.random.seed(args.randSeed)
    torch.manual_seed(args.randSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_num_threads(8)

    data_dir = './'
    isLds = True
    finetune = True
    channel_norm = True
    time_norm = False
    prepSxk = True
    stratified = False
    isFilt = False
    print('stratified', stratified)
    print('channel norm', channel_norm)

    hidden_dim = args.hidden_dim
    fs = 250;
    sec = 30
    # label_type = 'cls2'
    # label_type = 'cls9'

    timeLen = 1
    timeStep = 1
    filtLen = 1

    args.device = torch.device('cuda')
    torch.cuda.set_device(args.gpu_index)

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
        save_dir = './runs_srt/raw_24video_batch24_dataset_%s_timeLen5_tf16_sf16_multiFact2_lr0.000700_wd0.015000_epochs80_randSeed%d_fold10_cls2' % (
        args.dataset, args.randSeed)
    elif n_vids == 28:
        save_dir = './runs_srt/raw_28video_batch28_dataset_%s_timeLen5_fold10_%s_c0.100000_l1.000000' % (
        args.dataset, label_type)

    print(args)
    print(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset = args.dataset
    if dataset == 'first':
        n_subs = 61
    elif dataset == 'second':
        n_subs = 62
    elif dataset == 'both':
        n_subs = 123

    n_folds = 10
    n_per = round(n_subs / n_folds)

    val_method = args.val_method

    if val_method == '10_folds':
        val_fold = n_folds
    elif val_method == 'loo':
        val_fold = n_subs

    current_val = 0
    train_or_test = args.train_or_test
    subjects_results = np.zeros((n_subs, n_vids, sec))
    subjects_score = np.zeros(n_subs)

    if finetune:
        print('start finetuning')

        for base_long in [60]:
            use_features = 'features1_de_1s_lds.mat'
            print(use_features)

            results_finetune = {}
            results_finetune['train_loss_history'], results_finetune['val_loss_history'] = np.zeros(
                (val_fold, args.epochs_finetune)), np.zeros((val_fold, args.epochs_finetune))
            results_finetune['train_acc_history'], results_finetune['val_acc_history'] = np.zeros(
                (val_fold, args.epochs_finetune)), np.zeros((val_fold, args.epochs_finetune))
            results_finetune['best_val_acc'], results_finetune['best_val_loss'] = np.zeros(val_fold), np.zeros(val_fold)
            results_finetune['best_epoch'] = np.zeros(val_fold)
            results_finetune['best_confusion'] = np.zeros((val_fold, 9, 9))

            save_dir_ft = save_dir

            for fold in range(n_folds):
                print('fold :', fold)
                args.save_dir_ft = os.path.join(save_dir_ft, str(fold))

                pre_fold = fold
                print('pretrain fold', pre_fold)
                data_dir = os.path.join(save_dir, str(pre_fold), use_features)
                print(data_dir)
                data, label_repeat, n_samples = load_srt_pretrainFeat(data_dir, channel_norm, timeLen, timeStep, isFilt,
                                                                      filtLen, label_type)
                print('data loaded:', data.shape)
                # print('label repeat:', label_repeat)

                # np.random.seed(fold)
                if val_method == '10_folds':
                    iteration = [0]
                    if fold < n_folds - 1:
                        val_list = np.arange(n_per * fold, n_per * (fold + 1))
                    else:
                        val_list = np.arange(n_per * fold, n_subs)
                elif val_method == 'loo':
                    if fold < n_folds - 1:
                        val_list = np.arange(n_per * fold, n_per * (fold + 1))
                    else:
                        val_list = np.arange(n_per * fold, n_subs)
                    iteration = list(np.arange(0, len(list(val_list))))
                    print('val list', val_list)

                # whether to use special leave one out
                for iter in iteration:
                    if val_method == '10_folds':
                        val_sub = [int(val) for val in val_list]
                        print('current fold:', fold * n_per, 'current val_sub:', val_sub)
                        current_val = fold
                    elif val_method == 'loo':
                        val_sub = [val_list[int(iter)]]
                        print('current fold:', fold * n_per, 'current val_sub:', val_sub)
                        current_val = val_sub[0]

                    train_sub = np.array(list(set(np.arange(n_subs)) - set(val_sub)))
                    # train_sub = np.array([1])
                    print('val', val_sub)
                    print('train', train_sub)

                    data_train = data[list(train_sub), :, :].reshape(-1, data.shape[-1])
                    label_train = np.tile(label_repeat, len(train_sub)).reshape(-1)

                    # make sure the label_train and the label_val become 1 dim
                    print(data_train.shape, label_train.shape)
                    print('label_train length', len(label_train))

                    data_val = data[list(val_sub), :, :].reshape(-1, data.shape[-1])
                    # Select max channels
                    # data_val = data_val[:, chn_sel]
                    label_val = np.tile(label_repeat, len(val_sub)).reshape(-1)
                    print('label_val length', len(label_val))

                    trainset = DEDataset(data_train, label_train)
                    valset = DEDataset(data_val, label_val)
                    # print('data_val: ', data_val.shape)
                    # print('data train:', data_train.shape)

                    print(n_samples)

                    if stratified:
                        train_sampler = TrainSampler_sub(len(train_sub), n_samples=n_samples,
                                                         batch_size=args.batch_size_finetune, n_subs=9)
                        val_sampler = TrainSampler_sub(len(val_sub), n_samples=n_samples,
                                                       batch_size=args.batch_size_finetune, n_subs=9)

                        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, pin_memory=True,
                                                  num_workers=8)
                        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, pin_memory=True,
                                                num_workers=8)

                    else:
                        train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size_finetune, shuffle=True,
                                                  num_workers=8)
                        val_loader = DataLoader(dataset=valset, batch_size=args.batch_size_finetune, shuffle=False,
                                                num_workers=8)

                    inp_dim = data_train.shape[-1]
                    print('input dim:', inp_dim)

                    if label_type == 'cls9':
                        model = simpleNN3(inp_dim, hidden_dim, 9, 30, stratified).to(args.device)
                    elif label_type == 'cls3':
                        model = simpleNN3(inp_dim, hidden_dim, 3, 30, stratified).to(args.device)
                    elif label_type == 'cls2':
                        model = simpleNN3(inp_dim, hidden_dim, 2, 30, stratified).to(args.device)

                    # model = ConvNet_debug(n_spatialFilters, n_timeFilters, timeFilterLen, hidden_dim, 2, n_channs, data_len).to(args.device)
                    print(model)

                    if train_or_test == 'train':

                        para_num = sum([p.data.nelement() for p in model.parameters()])
                        print('Total number of parameters:', para_num)

                        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate_finetune,
                                                     weight_decay=0.05)

                        print('save_dir_ft: ', save_dir_ft)

                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs_finetune,
                                                                    gamma=0.8,
                                                                    last_epoch=-1, verbose=False)
                        criterion = torch.nn.CrossEntropyLoss().to(args.device)

                        best_epoch, train_loss_history, val_loss_history, train_acc_history, val_acc_history, best_confusion = train_earlyStopping(
                            args, train_loader, val_loader, model, criterion, optimizer, scheduler, True)

                        print('Current val:', current_val)
                        results_finetune['train_loss_history'][current_val, :], results_finetune['val_loss_history'][
                                                                                current_val,
                                                                                :] = train_loss_history, val_loss_history
                        results_finetune['train_acc_history'][current_val, :], results_finetune['val_acc_history'][
                                                                               current_val,
                                                                               :] = train_acc_history, val_acc_history
                        results_finetune['best_val_acc'][current_val] = results_finetune['val_acc_history'][
                            current_val, best_epoch]
                        results_finetune['best_val_loss'][current_val] = results_finetune['val_loss_history'][
                            current_val, best_epoch]
                        results_finetune['best_epoch'][current_val] = best_epoch
                        results_finetune['best_confusion'][current_val, :, :] = best_confusion

                        # # You can choose to save as one file or many files
                        # with open(os.path.join(save_dir_ft, val_method + '_' + str(
                        #         fold) + '_' + args.dataset + '_dataset_results_finetune.pkl'), 'wb') as f:
                        #     pickle.dump(results_finetune, f)

                    elif train_or_test == 'test':  # HERE MEANS TEST FOR 1-fold in the k-fold process

                        best_epoch_file = os.path.join(save_dir_ft,
                                                       val_method + '_' + args.dataset + '_dataset_results_finetune.pkl')
                        f = open(best_epoch_file, 'rb')
                        best_epoch_list = pickle.load(f)['best_epoch']
                        print('Shape of the best epoch list :', len(best_epoch_list))

                        best_epoch = int(best_epoch_list[current_val])
                        if int(best_epoch) < 10:
                            best_epoch = '0' + str(best_epoch)
                        else:
                            best_epoch = str(best_epoch)
                        print('Epoch :', best_epoch, ' in fold:', str(fold), ' for val:', str(current_val))
                        # U should choose another file once you change the hyperparameter of the finetune section
                        state_dict = os.path.join(save_dir_ft, str(fold),
                                                  'finetune_checkpoint_00%s.pth.tar' % (best_epoch))
                        model.load_state_dict(torch.load(state_dict, map_location=args.device)['state_dict'])
                        model.eval()
                        test_acc = 0
                        results = []

                        # validation
                        for counter, (x_batch, y_batch) in enumerate(val_loader):
                            x_batch = x_batch.to(args.device)
                            y_batch = y_batch.to(args.device)
                            logits = model(x_batch)
                            _, result = torch.max(logits, dim=1)
                            results.extend(list(result.cpu().numpy()))

                        print('Test mode, val_sub:', val_sub)

                        subjects_results[val_sub, :, :] = np.array(results).reshape(-1, n_vids, sec)

                        label_repeat = np.array(label_repeat).reshape(n_vids, sec)
                        for sub in val_sub:
                            # average the 30s score of 28 videos
                            subjects_score[sub] = np.mean(
                                [np.sum(subjects_results[sub, vid, :] == label_repeat[vid, :]) /
                                 subjects_results.shape[-1] for vid in range(0, n_vids)])

            # Save the test process results
            if train_or_test == 'test':
                mlp = {'mlp': subjects_results}
                sio.savemat('./Clisa_results_%s_%s.mat' % (label_type, val_method), mlp)
                pd.DataFrame(subjects_score).to_csv('./Clisa_score_%s_%s.csv' % (label_type, val_method))
                print('Save the val results.')


            elif train_or_test == 'train':
                print(save_dir_ft)
                print(args)
                print('val loss mean: %.3f, std: %.3f; val acc mean: %.3f, std: %.3f' % (
                    np.mean(results_finetune['best_val_loss']), np.std(results_finetune['best_val_loss']),
                    np.mean(results_finetune['best_val_acc']), np.std(results_finetune['best_val_acc']))
                      )
                with open(os.path.join(save_dir_ft, val_method + '_' + args.dataset + '_dataset_results_finetune.pkl'),
                          'wb') as f:
                    pickle.dump(results_finetune, f)
