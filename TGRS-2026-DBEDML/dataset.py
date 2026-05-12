import math
import torch
import scipy.io as sio
import random
import numpy as np
from sklearn import preprocessing
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

def sanity_check(all_set):
    nclass = 0
    nsamples = 0
    all_good = {}
    for class_ in all_set:
        if len(all_set[class_]) >= 200:
            all_good[class_] = all_set[class_][:200]
            nclass += 1
            nsamples += len(all_good[class_])
    return all_good


def load_data(image_file, label_file, args):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    if args.dataset == 'PaviaU':
        data_key = 'paviaU'
        label_key = 'paviaU_gt'
    elif args.dataset == 'Yancheng':
        data_key = 'data'
        label_key = 'groundtruth'
    else:
        data_key = 'data'
        label_key = 'label'
    data_all = image_data[data_key].astype(np.float32)  # dic-> narray , KSC:ndarray(512,217,204)
    GroundTruth = label_data[label_key].astype(np.int64)
    if args.dataset == 'Willow':
        mask_3 = (GroundTruth == 3)
        mask_9 = (GroundTruth == 9)
        GroundTruth[mask_3] = 9
        GroundTruth[mask_9] = 3
    elif args.dataset == 'Robinia':
        mask_7 = (GroundTruth == 7)
        mask_11 = (GroundTruth == 11)
        GroundTruth[mask_7] = 11
        GroundTruth[mask_11] = 7#刺槐林，替换的是第7类
    elif args.dataset == 'Tamarix':
        mask_7 = (GroundTruth == 3)
        mask_9 = (GroundTruth == 9)
        GroundTruth[mask_7] = 9
        GroundTruth[mask_9] = 3
    elif args.dataset == 'Yancheng':
        mask_7 = (GroundTruth == 11)
        mask_9 = (GroundTruth == 18)
        GroundTruth[mask_7] = 18
        GroundTruth[mask_9] = 11
    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    return Data_Band_Scaler, GroundTruth  # image:(512,217,3),label:(512,217)


def flip(data):
    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data


class matcifar(data.Dataset):
    def __init__(self, imdb, train, d, medicinal):

        self.train = train  # training set or test set
        self.imdb = imdb
        self.d = d
        self.x1 = np.argwhere(self.imdb['set'] == 1)
        self.x2 = np.argwhere(self.imdb['set'] == 3)
        self.x1 = self.x1.flatten()
        self.x2 = self.x2.flatten()
        if medicinal == 1:
            self.train_data = self.imdb['data'][self.x1, :, :, :]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][self.x2, :, :, :]
            self.test_labels = self.imdb['Labels'][self.x2]

        else:
            self.train_data = self.imdb['data'][:, :, :, self.x1]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][:, :, :, self.x2]
            self.test_labels = self.imdb['Labels'][self.x2]
            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))  ##(17, 17, 200, 10249)
                self.test_data = self.test_data.transpose((3, 2, 0, 1))
            else:
                self.train_data = self.train_data.transpose((3, 0, 2, 1))
                self.test_data = self.test_data.transpose((3, 0, 2, 1))

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:

            img, target = self.test_data[index], self.test_labels[index]
        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, args):
    Data_Band_Scaler = Data_Band_Scaler[:,:,:args.spectral_size]
    nRow, nColumn, nBand = Data_Band_Scaler.shape
    data_band_scaler = flip(Data_Band_Scaler)
    groundtruth = flip(GroundTruth)
    HalfWidth = (args.patch - 1) // 2
    G = groundtruth[
        nRow - HalfWidth : 2 * nRow + HalfWidth,
        nColumn - HalfWidth : 2 * nColumn + HalfWidth
    ]
    data = data_band_scaler[
        nRow - HalfWidth : 2 * nRow + HalfWidth,
        nColumn - HalfWidth : 2 * nColumn + HalfWidth, :
    ]
    
    [Row, Column] = np.where(G[4:-4,4:-4]>=0)
    Row += HalfWidth
    Column += HalfWidth
    
    train, test, da_train, DA_train = {}, {}, {}, {}
    m = np.max(G).astype(int)
    nlabeled = shot_num_per_class
    for i in range(m):
        condition = i + 1 in args.known_classes
        indices = [
            j for j, _ in enumerate(Row.ravel().tolist())
            if G[Row[j], Column[j]] == i + 1
        ]
        nb_val = shot_num_per_class
        np.random.shuffle(indices)
        DA_train[i] = indices[:nb_val] * (
                math.ceil((200 - nlabeled) / nlabeled) + 1
            )
        if condition:
            train[i] = indices[:nb_val]
            da_train[i] = indices[:nb_val] * (
                math.ceil((200 - nlabeled) / nlabeled) + 1
            )
            test[i] = indices[nb_val:]
        else:
            test[i] = indices
        #print('------------',len(test[i]))
        '''
        if i == 0:
            indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == 0]
            print(len(indices),'+++++++++++++++++')
            np.random.shuffle(indices)
            test[i] = test[i] + indices
            #print('##########',len(test[i]))
        '''
    train_indices = sum(
        [train[i] for i in range(m) if i + 1 in args.known_classes], []
    )
    test_indices = sum(
        [test[i] for i in range(m)], []
    )
    da_train_indices = sum(
        [da_train[i] for i in range(m) if i + 1 in args.known_classes], []
    )
    DA_train_indices = sum(
        [DA_train[i] for i in range(m)], []
    )
    np.random.shuffle(test_indices)
    x_test = [Row[i] - HalfWidth for i in test_indices]
    y_test = [Column[i] - HalfWidth for i in test_indices]
    x_train = [Row[i] - HalfWidth for i in train_indices]
    y_train = [Column[i] - HalfWidth for i in train_indices]
    indices = [x_test, y_test, x_train, y_train]
    nTrain, nTest, da_nTrain, DA_nTrain = len(train_indices), len(test_indices), len(da_train_indices), len(DA_train_indices)
    '''*******************************************************************'''
    imdb = {
        'data': np.zeros(
            [2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest],
            dtype=np.float32
        ),
        'Labels': np.zeros(nTrain + nTest, dtype=np.int64),
        'set': np.zeros(nTrain + nTest, dtype=np.int64)
    }
    RandPerm = np.array(train_indices + test_indices)
    for iSample in range(nTrain + nTest):
        r, c = Row[RandPerm[iSample]], Column[RandPerm[iSample]]
        imdb['data'][:, :, :, iSample] = data[
            r - HalfWidth : r + HalfWidth + 1,
            c - HalfWidth : c + HalfWidth + 1, :
        ]
        imdb['Labels'][iSample] = G[r, c].astype(np.int64)
    
    imdb['Labels'] -= 1
    imdb['set'] = np.concatenate([
        np.ones(nTrain), 3 * np.ones(nTest)
    ]).astype(np.int64)
    train_dataset = matcifar(imdb, True, 3, 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=class_num * shot_num_per_class,
        shuffle=False,
        num_workers=0
    )
    test_dataset = matcifar(imdb, False, 3, 0)
    test_loader = DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0
    )
    '''*******************************************************************'''
    imdb_da_train = {
        'data': np.zeros(
            [2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],
            dtype=np.float32
        ),
        'Labels': np.zeros(da_nTrain, dtype=np.int64),
        'set': np.ones(da_nTrain, dtype=np.int64)
    }
    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):
        r, c = Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]
        patch = data[
            r - HalfWidth : r + HalfWidth + 1,
            c - HalfWidth : c + HalfWidth + 1, :
        ]
        imdb_da_train['data'][:, :, :, iSample] = radiation_noise(patch)
        imdb_da_train['Labels'][iSample] = G[r, c].astype(np.int64)
    imdb_da_train['Labels'] += 60
    '''*******************************************************************'''
    imdb_DA_train = {
        'data': np.zeros(
            [2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, DA_nTrain],
            dtype=np.float32
        ),
        'Labels': np.zeros(DA_nTrain, dtype=np.int64),
        'set': np.ones(DA_nTrain, dtype=np.int64)
    }
    DA_RandPerm = np.array(DA_train_indices)
    for iSample in range(DA_nTrain):
        r, c = Row[DA_RandPerm[iSample]], Column[DA_RandPerm[iSample]]
        patch = data[
            r - HalfWidth : r + HalfWidth + 1,
            c - HalfWidth : c + HalfWidth + 1, :
        ]
        imdb_DA_train['data'][:, :, :, iSample] = radiation_noise(patch)
        imdb_DA_train['Labels'][iSample] = G[r, c].astype(np.int64)

    return train_loader, test_loader, imdb_da_train, imdb_DA_train, indices
def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, args):
    train_loader, test_loader, imdb_da_train, imdb_DA_train, indice = get_train_test_loader(#imdb_DA_train
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth,
        class_num=class_num,
        shot_num_per_class=shot_num_per_class,
        args=args
    )
    del Data_Band_Scaler
    del GroundTruth
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))
    target_da_labels = imdb_da_train['Labels']
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    target_dataset = matcifar(imdb_DA_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=160, shuffle=True, num_workers=0, drop_last=True)
    del imdb_DA_train
    del imdb_da_train
    del target_da_labels
    del target_da_datas
    return train_loader, test_loader, target_da_train_set, target_loader, indice

class Traindata:
    def __init__(self, dataset, config):
        self.dset = dataset
        self.n_closed = config.n_ways
        self.n_open = config.n_open_ways
        self.shots = config.n_shots
        self.queries = config.n_queries
        self.iterations = config.episodes
        self.class_labels = sorted(dataset.keys())
        self.spectral_size = config.spectral_size
    def get_episode_0(self):
        return self._build_episode(
            self.class_labels[:len(self.class_labels)-self.n_closed],
            self.class_labels[len(self.class_labels)-self.n_closed:]
        )
    def get_episode_1(self):
        return self._build_episode(
            self.class_labels[len(self.class_labels)-self.n_closed:],
            self.class_labels[:len(self.class_labels)-self.n_closed]
        )
    
    def _build_episode(self, closed_cls, open_cls):
        def process_class(cls_group, is_open=False):
            support_data, support_labels = [], []
            query_data, query_labels = [], []
            
            for idx, cls_id in enumerate(np.random.permutation(cls_group)[:self.n_closed if not is_open else self.n_open]):
                samples = self.dset[cls_id]
                support_indices = np.random.permutation(len(samples))[:self.shots]
                
                support_data += [samples[i] for i in support_indices]
                support_labels += [idx] * self.shots
                
                query_indices = np.setdiff1d(np.arange(len(samples)), support_indices)
                query_indices = np.random.permutation(query_indices)[:self.queries]
                
                query_data += [samples[i] for i in query_indices]
                query_labels += [cls_id if is_open else idx] * self.queries
            
            return support_data, support_labels, query_data, query_labels
        
        closed_support, closed_support_labels, closed_query, closed_query_labels = process_class(closed_cls)
        open_support, open_support_labels, open_query, open_query_labels = process_class(open_cls, True)
        
        def tensor_wrap(data, labels):
            data_tensor = torch.stack([torch.as_tensor(d[:self.spectral_size]) for d in data])
            label_tensor = torch.tensor(labels)
            return data_tensor.unsqueeze(0), label_tensor.unsqueeze(0)
        
        closed_support_t, closed_support_l_t = tensor_wrap(closed_support, closed_support_labels)
        closed_query_t, closed_query_l_t = tensor_wrap(closed_query, closed_query_labels)
        open_support_t, open_support_l_t = tensor_wrap(open_support, open_support_labels)
        open_query_t, open_query_l_t = tensor_wrap(open_query, open_query_labels)
        
        return (
            closed_support_t, closed_support_l_t,
            closed_query_t, closed_query_l_t,
            open_support_t, open_support_l_t,
            open_query_t, open_query_l_t
        )
    
    def generate_random_episode(self, idx=None):
        selector = 0 if idx is None or idx % 2 == 0 else 1
        return [self.get_episode_0, self.get_episode_1][selector]()