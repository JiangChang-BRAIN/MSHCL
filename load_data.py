import numpy as np
import scipy.io as sio
from io_utils import smooth_moving_average
import os
import h5py
import pickle


def load_srt_raw_newPre(timeLen, timeStep, fs, channel_norm, time_norm, label_type):
    n_channs = 30
    n_points = 7500
    data_len = fs * timeLen
    n_segs = int((n_points/fs - timeLen) / timeStep + 1)
    print('n_segs:', n_segs)

    data_path = 'Clisa_data'
    data_paths = os.listdir(data_path)
    data_paths.sort()
    n_vids = 28; chn = 30; fs = 250; sec = 30;

    data = np.zeros((len(data_paths),n_vids,chn, fs * sec))

    for idx, path in enumerate(data_paths):
        with open(os.path.join(data_path, path), 'rb') as f:
            data_sub = pickle.load(f)
            data[idx, :, :, :] = data_sub[:,:-2,:]

    # data shape :(sub, vid, chn, fs * sec)
    print('data loaded:', data.shape)

    n_subs = data.shape[0]

    # Only use positive and negative samples
    if label_type == 'cls2':
        vid_sel = list(range(12))
        vid_sel.extend(list(range(16,28)))
        data = data[:, vid_sel, :, :] # sub, vid, n_channs, n_points
        n_videos = 24
    else:
        n_videos = 28

    print('classification:', label_type)

    data = np.transpose(data, (0,1,3,2)).reshape(n_subs, -1, n_channs)

    if channel_norm:
        for i in range(data.shape[0]):
            data[i,:,:] = (data[i,:,:] - np.mean(data[i,:,:], axis=0)) / np.std(data[i,:,:], axis=0)

    if time_norm:
        data = (data - np.tile(np.expand_dims(np.mean(data, axis=2), 2), (1, 1, data.shape[2]))) / np.tile(
            np.expand_dims(np.std(data, axis=2), 2), (1, 1, data.shape[2])
        )

    n_samples = np.ones(n_videos)*n_segs

    if label_type == 'cls2':
        label = [0] * 12
        label.extend([1] * 12)

    elif label_type == 'cls9':
        label = [0] * 3
        for i in range(1,4):
            label.extend([i] * 3)
        label.extend([4] * 4)
        for i in range(5,9):
            label.extend([i] * 3)
        print(label)

    elif label_type == 'cls3':
        label = [0] * 12
        label.extend([1] * 4)
        label.extend([2] * 12)

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]]*n_segs

    return data, label_repeat, n_samples, n_segs



def load_srt_pretrainFeat(datadir, channel_norm, timeLen, timeStep, isFilt, filtLen, label_type):
    if label_type == 'cls2':
        n_samples = np.ones(24).astype(np.int32) * 30
    else:
        n_samples = np.ones(28).astype(np.int32) * 30

    for i in range(len(n_samples)):
        n_samples[i] = int((n_samples[i] - timeLen) / timeStep + 1)

    if datadir[-4:] == '.npy':
        data = np.load(datadir)
        data[data < -10] = -5
    elif datadir[-4:] == '.mat':
        data = sio.loadmat(datadir)['de_lds']
        print('isnan total:', np.sum(np.isnan(data)))
        data[np.isnan(data)] = -8
        # data[data < -8] = -8
    
    # data_use = data[:, np.max(data, axis=0)>1e-6]
    # data = data.reshape(45, int(np.sum(n_samples)), 256)
    print(data.shape)
    print(np.min(data), np.median(data))

    n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples)))
    if isFilt:
        print('filtLen', filtLen)
        data = data.transpose(0,2,1)
        for i in range(data.shape[0]):
            for vid in range(len(n_samples)):
                data[i, :, int(n_samples_cum[vid]): int(n_samples_cum[vid+1])] = smooth_moving_average(data[
                                            i, :, int(n_samples_cum[vid]): int(n_samples_cum[vid+1])], filtLen)
        data = data.transpose(0,2,1)

    # Normalization for each sub
    if channel_norm:
        print('subtract mean and divided by var')
        for i in range(data.shape[0]):
            #data[i,:,:] = data[i,:,:] - np.mean(data[i,:,:], axis=0)
            data[i,:,:] = (data[i,:,:] - np.mean(data[i,:,:], axis=0)) / (np.std(data[i,:,:], axis=0) + 1e-3)

    if label_type == 'cls2':
        label = [0] * 12
        label.extend([1] * 12)

    elif label_type == 'cls9':
        label = [0] * 3
        for i in range(1,4):
            label.extend([i] * 3)
        label.extend([4] * 4)
        for i in range(5,9):
            label.extend([i] * 3)
        print(label)

    elif label_type == 'cls3':
        label = [0] * 12
        label.extend([1] * 4)
        label.extend([2] * 12)
        print(label)

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]]*n_samples[i]
    return data, label_repeat, n_samples


