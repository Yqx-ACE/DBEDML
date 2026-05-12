import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import argparse
import pickle
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import util,draw
from tqdm import tqdm
from torch.utils.data import DataLoader
from util import seed_torch, adjust_learning_rate
from NetworkPre import FeatureNet
from dataset import sanity_check, load_data, get_target_dataset, Traindata
from torchmetrics import Accuracy
from sklearn.metrics import precision_score, classification_report, cohen_kappa_score, confusion_matrix, recall_score
from torch.optim import Adam
from dataset import matcifar
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR
from thop import profile
import warnings
warnings.filterwarnings("ignore")
torch.set_printoptions(profile="full")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())
def parse_args():
    parser = argparse.ArgumentParser(description="CDFSOSR_for_HSI_Classification")
    parser.add_argument('--dataset', type=str, default='PaviaU', choices = ['PaviaU','Willow', 'Robinia','Tamarix', 'Yancheng'])
    parser.add_argument("-z", "--test_lsample_num_per_class", type=int, default=5)
    parser.add_argument('--patch', type=int, default=9, choices=[7, 9, 11, 13, 15, 17])
    parser.add_argument('--n_shots', type=int, default=1)
    parser.add_argument('--n_queries', type=int, default=19)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--emd_dim', type=int, default=64)
    parser.add_argument('--fea_dim', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--lr_decay_epochs', type=str, default='100,200')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--funit', type=float, default=1.0)
    parser.add_argument('--m_in', type=float, default=-1)
    parser.add_argument('--m_out', type=float, default=1)
    parser.add_argument('--ahead_combine', action='store_true', default=True)
    parser.add_argument('--learnable_margin', action='store_true', default=False)
    parser.add_argument('--top_method', type=str, default='all', choices=['query', 'proto', 'all'])
    parser.add_argument('--energy_method', type=str, default="sum", choices=["sum", "min"])
    parser.add_argument('--weighted_combine', type=str, default=False)
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--fusion_weight',default = 0.40, type=float)
    args = parser.parse_args()
    if args.dataset == 'PaviaU':
        args.known_classes = [1, 2, 3, 4, 5, 6, 7, 8]
        args.unknown_classes = [9]
        args.n_ways = 8
        args.spectral_size = 103
        args.n_open_ways = args.n_ways
    elif args.dataset == 'Willow':
        args.known_classes = [1, 2, 3, 4, 5, 6, 7, 8,]
        args.unknown_classes = [9, ]
        args.n_ways = 8
        args.n_open_ways = args.n_ways
        args.spectral_size = 126
    elif args.dataset == 'Robinia':
        args.known_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        args.unknown_classes = [11, ]
        args.n_ways = 10
        args.n_open_ways = args.n_ways
        args.spectral_size = 126
    elif args.dataset == 'Tamarix':
        args.known_classes = [1, 2, 3, 4, 5, 6, 7, 8,]
        args.unknown_classes = [9, ]
        args.n_ways = 8
        args.n_open_ways = args.n_ways
        args.spectral_size = 126 
    elif args.dataset == 'Yancheng':
        args.known_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, ]
        args.unknown_classes = [18, ]
        args.n_ways = 17
        args.n_open_ways = args.n_ways
        args.spectral_size = 128
    return args

def dataload(args):
    with open(os.path.join('/mnt/hdd/yqx/DCFSL-2021-main/datasets',  'Chikusei_imdb_128.pickle'), 'rb') as handle:
        source_imdb = pickle.load(handle)
    source_imdb['data'] = np.array(source_imdb['data'])
    source_imdb['Labels'] = np.array(source_imdb['Labels'], dtype='int')
    source_imdb['set'] = np.array(source_imdb['set'], dtype='int')
    keys_all_train = sorted(list(set(source_imdb['Labels'])))
    label_encoder_train = {}
    for i in range(len(keys_all_train)):
        label_encoder_train[keys_all_train[i]] = i
    train_set = {}
    for class_, path in zip(source_imdb['Labels'], source_imdb['data']):
        encoded_label = label_encoder_train[class_]
        if encoded_label not in train_set:
            train_set[encoded_label] = []
        train_set[encoded_label].append(path)
    data = sanity_check(train_set)
    del train_set
    source_imdb['data'] = source_imdb['data'][:,:,:,:args.spectral_size].transpose((1, 2, 3, 0))
    source_dataset = matcifar(source_imdb, train=True, d=3, medicinal=0)
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=160, shuffle=True, num_workers=0, drop_last=True)
    del source_imdb
    for class_ in data:
        for i in range(len(data[class_])):
            data[class_][i] = np.transpose(data[class_][i], (2, 0, 1))
    if args.dataset == 'PaviaU':
        test_data_path = os.path.join('/mnt/hdd/yqx/DCFSL-2021-main/datasets/paviaU/Indian Pines', args.dataset + '.mat')
        test_label_path = os.path.join('/mnt/hdd/yqx/DCFSL-2021-main/datasets/paviaU/Indian Pines',  args.dataset + '_gt.mat')
    elif args.dataset == 'Willow':
        test_data_path = os.path.join("/mnt/hdd/yqx/DCFSL-2021-main/datasets/22liulin_pre1.mat")
        test_label_path = os.path.join("/mnt/hdd/yqx/DCFSL-2021-main/datasets/22liulin_gt.mat")
    elif args.dataset == 'Robinia':
        test_data_path = os.path.join("/mnt/hdd/yqx/DCFSL-2021-main/datasets/cihuai.mat")
        test_label_path = os.path.join("/mnt/hdd/yqx/DCFSL-2021-main/datasets/cihuai_gt.mat")
    elif args.dataset == 'Tamarix':
        test_data_path = os.path.join("/mnt/hdd/yqx/DCFSL-2021-main/datasets/9chengliu_pre1.mat")
        test_label_path = os.path.join("/mnt/hdd/yqx/DCFSL-2021-main/datasets/9chengliu_label.mat")
    elif args.dataset == 'Yancheng':
        test_data_path = os.path.join("/mnt/hdd/yqx/DCFSL-2021-main/datasets/yancheng.mat")
        test_label_path = os.path.join("/mnt/hdd/yqx/DCFSL-2021-main/datasets/yancheng_gt.mat")
    Data_Band_Scaler, GroundTruth = load_data(test_data_path, test_label_path, args)
    test_known_loader, test_loader, target_da_metatrain_data, target_loader, indice = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth,
        class_num=args.n_ways,
        shot_num_per_class=args.test_lsample_num_per_class,
        args=args
    )
    train_data = dict(data)
    train_data.update(target_da_metatrain_data)
    return train_data, test_known_loader, test_loader, source_loader, target_loader, GroundTruth, indice

def train(args, model, train_loader,test_known_loader, test_loader, source_loader, target_loader, seed):
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = [int(it) for it in iterations]
    dataset = Traindata(train_loader, args)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    train_acc_meter = Accuracy(task="multiclass", num_classes=args.n_ways).to(device=0)
    train_open_acc_meter = Accuracy(task="multiclass", num_classes=args.n_ways + 1).to(device=0)
    model.train()
    best_oa, best_kappa, best_aa, C, best_episodes, test_begin, test_end, best_prediction, best_result  = 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0, None, ''
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    for episode in range(args.episodes):
        try:
            source_data, source_label = source_iter.__next__()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.__next__()
        try:
            target_data, target_label = target_iter.__next__()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.__next__()
        if episode % 2 == 0:
            data = dataset.get_episode_1()
            support_data, support_label, query_data, query_label, suppopen_data, suppopen_label, openset_data, openset_label = data
            support_data = support_data.float().cuda()
            support_label = support_label.cuda().long()
            query_data = query_data.float().cuda()
            query_label = query_label.cuda().long()
            suppopen_data = suppopen_data.float().cuda()
            suppopen_label = suppopen_label.cuda().long()
            openset_data = openset_data.float().cuda()
            openset_label = openset_label.cuda().long()
            target_data = target_data.float().cuda().unsqueeze(0)
            target_label = target_label.cuda().long()
            openset_label = args.n_ways * torch.ones_like(openset_label)
            the_img = (support_data,query_data,suppopen_data,openset_data,target_data)
            the_label = (support_label,query_label,suppopen_label,openset_label,target_label)
            probs, loss = model(the_img, the_label)
            (loss_cls, loss_funit), loss_da = loss
            total_loss = loss_funit + loss_cls + loss_da
            model.zero_grad()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            close_pred = torch.argmax(probs[0][:, :, :args.n_ways].view(-1, args.n_ways), -1)
            close_label = query_label.view(-1)
            open_pred = torch.argmax(torch.cat((probs[0].view(-1, args.n_ways + 1), probs[1].view(-1, args.n_ways + 1)), dim=0),-1)
            open_label = torch.cat((query_label.view(-1), openset_label.view(-1)))
            train_acc_meter.update(close_pred, close_label)
            train_open_acc_meter.update(open_pred, open_label)
        else:
            data = dataset.get_episode_0()
            support_data, support_label, query_data, query_label, suppopen_data, suppopen_label, openset_data, openset_label = data
            support_data = support_data.float().cuda()
            support_label = support_label.cuda().long()
            query_data = query_data.float().cuda()
            query_label = query_label.cuda().long()
            suppopen_data = suppopen_data.float().cuda()
            suppopen_label = suppopen_label.cuda().long()
            openset_data = openset_data.float().cuda()
            openset_label = openset_label.cuda().long()
            source_data = source_data.float().cuda().unsqueeze(0)
            source_label = source_label.cuda().long()
            openset_label = args.n_ways * torch.ones_like(openset_label)
            the_img = (support_data, query_data, suppopen_data, openset_data, source_data)
            the_label = (support_label, query_label, suppopen_label, openset_label, source_label)
            probs, loss = model(the_img, the_label)
            (loss_cls, loss_funit), _ = loss
            total_loss = loss_funit + loss_cls
            model.zero_grad()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            close_pred = torch.argmax(probs[0][:, :, :args.n_ways].view(-1, args.n_ways), -1)
            close_label = query_label.view(-1)
            open_pred = torch.argmax(torch.cat((probs[0].view(-1, args.n_ways + 1), probs[1].view(-1, args.n_ways + 1)), dim=0),-1)
            open_label = torch.cat((query_label.view(-1), openset_label.view(-1)))
            train_acc_meter.update(close_pred, close_label)
            train_open_acc_meter.update(open_pred, open_label)
        if (episode + 1) % 10000 == 0:
            print('episode {:>3d}:,   ACC: {:6.4f}, Open_ACC: {:6.4f}'.format(episode + 1, train_acc_meter.compute().item(), train_open_acc_meter.compute().item()))
        if (episode + 1) % 100  == 0:
            test_begin = time.time() 
            kappa, oa, aa, C, prediction, label, result = test(args, model, test_known_loader, test_loader, episode + 1)
            test_end = time.time()
            model.train()
            if(oa > best_oa):
                best_oa = oa
                best_kappa = kappa
                best_aa = aa
                best_C = C
                best_episodes = episode + 1
                best_prediction = prediction
                best_result = result
            train_acc_meter.reset()
            train_open_acc_meter.reset()
            print('episode {:>3d}:,   ACC: {:6.4f}'.format(episode + 1, oa * 100))
            torch.save(model.state_dict(),"/mnt/hdd/yqx/Cross_Domain_FSL_Open_Set_Recognition/PKL/"+str(args.dataset + '_' + str(seed) + ".pkl"))
    print('Best_OA is:',best_oa)
    print('Best_episodes is:',best_episodes)
    return best_oa, best_aa, best_kappa, best_C, test_end - test_begin, best_prediction, label, best_result
def test(args, model, test_known_loader, test_loader, episodes):
    model.eval()
    torch.cuda.empty_cache()
    prediction = []
    label = []
    support_data, support_label = next(iter(test_known_loader))
    support_data = torch.unsqueeze(support_data, dim=0).float().cuda()
    support_label = torch.unsqueeze(support_label, dim=0).cuda().long()
    with tqdm(test_loader, total=len(test_loader), leave=False) as pbar:
        for idx, data in enumerate(pbar):
            query_data, query_label = data
            query_data = torch.unsqueeze(query_data, dim=0).float().cuda()
            query_label = torch.unsqueeze(query_label, dim=0).cuda().long()
            the_img = (support_data, query_data, support_data, query_data, query_data)
            the_label = (support_label, query_label, support_label, query_label, query_label)
            probs = model(the_img, the_label, is_test=True)
            pred = torch.argmax(probs.view(-1, args.n_ways + 1), -1)
            prediction.append(pred.cpu().detach().numpy())
            label.append(query_label.view(-1).cpu().detach().numpy())
    label = np.concatenate(label)
    prediction = np.concatenate(prediction)
    kappa = cohen_kappa_score(label, prediction)
    oa = precision_score(label, prediction, average="micro")
    aa = recall_score(label, prediction, average="macro")
    C = confusion_matrix(label, prediction)
    all_result = classification_report(label, prediction, digits=4)
    return kappa, oa, aa, C, prediction, label, all_result

import time
from datetime import datetime

def main(seed):
    args = parse_args()
    train_loader, test_known_loader, test_loader, source_loader, target_loader, gt, indice = dataload(args)
    model = FeatureNet(args.n_ways, args.emd_dim, args.fea_dim, args)
    model.cuda()
    train_begin = time.time()
    best_oa, best_aa, best_kappa, best_C, test_time, best_prediction, label, best_result = train(args, model, train_loader, test_known_loader, test_loader, source_loader, target_loader, seed)
    #util.get_cls_map(best_prediction, label, gt, indice, best_oa, args)
    #print(best_result)
    train_end = time.time()
    return best_oa, best_aa, best_kappa, best_C, train_end-train_begin, test_time

def save_evaluation_results(
    OA_,
    OAMean, OAStd, 
    AAMean, AAStd, 
    kMean, kStd, 
    AMean, AStd, 
    train_time, test_time,
    args, 
    output_dir="/mnt/hdd/yqx/Cross_Domain_FSL_Open_Set_Recognition/result/", 
    prefix="evaluation", 
    include_console=True
):
    prefix = args.dataset if args else prefix
    OA = "{:.2f}".format(OAMean * 100)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_path = f"{output_dir.rstrip('/')}/{prefix}_{timestamp}_{OA}.txt"
    with open(file_path, 'w') as f:
        def output(content):
            if include_console:
                print(content)
            f.write(content + '\n')
        
        if args is not None:
            output(f"\n{'='*20} 命令行参数 {'='*20}")
            for arg, value in vars(args).items():
                output(f"{arg}: {value}")
        
        output(f"\n{'='*20} 评估指标 {'='*20}")
        output("Train time: {:.2f}".format(train_time))
        output("Test time: {:.2f}".format(test_time))
        output("Average OA: {:.2f} +- {:.2f}".format(100 * OAMean, 100 * OAStd))
        output("Average AA: {:.2f} +- {:.2f}".format(100 * AAMean, 100 * AAStd))
        output("Average Kappa: {:.2f} +- {:.2f}".format(100 * kMean, 100 * kStd))
        
        output("Accuracy for each class: ")
        for i in range(args.n_ways + 1 if args else 0):
            output("Class {}: {:.2f} +- {:.2f}".format(i, 100 * AMean[i], 100 * AStd[i]))
        output("Overall Accuracy for each running: ")
        for i in range(len(OA_)):
            output("The {}: {:.2f}".format(i, 100 * OA_[i][0]))
    if include_console:
        print(f"评估结果已保存至: {file_path}")
    
    return file_path

if __name__ == "__main__":
    args = parse_args()
    nDataSet = 10
    seedx = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]
    OA = np.zeros([nDataSet, 1])
    A = np.zeros([nDataSet, args.n_ways + 1])
    Kappa = np.zeros([nDataSet, 1])
    for i in range(nDataSet):
        print('-------------------------',i+1,'-----------------------------')
        torch.cuda.empty_cache()
        seed = seedx[i]
        seed_torch(seed)
        best_oa, best_aa, best_kappa, best_C, train_time, test_time = main(seed)
        print('Train_time:',train_time)
        print('Test_time:',test_time)
        OA[i] = best_oa
        A[i,:] = np.diag(best_C) / np.sum(best_C, 1, dtype=np.float32)
        Kappa[i] = best_kappa
    AA = np.mean(A,1)
    AAMean = np.mean(AA,0)
    AAStd = np.std(AA)
    AMean = np.mean(A, 0)
    AStd = np.std(A, 0)
    OAMean = np.mean(OA)
    OAStd = np.std(OA)
    kMean = np.mean(Kappa)
    kStd = np.std(Kappa)
    save_evaluation_results(OA, OAMean, OAStd, AAMean, AAStd, kMean, kStd, AMean, AStd, train_time, test_time, args)