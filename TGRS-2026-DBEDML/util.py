import os
import torch
import random
import numpy as np
import json

import matplotlib.pyplot as plt

def get_classification_map(y_pred, y, support_label, indices):
    x_test, y_test, x_train, y_train = indices
    height = y.shape[0]
    width = y.shape[1]
    cls_labels = np.zeros((height, width))
    for i in range(len(x_test)):
        cls_labels[x_test[i], y_test[i]]=y_pred[i]
    for i in range(len(x_train)):
        cls_labels[x_train[i], y_train[i]]=y_pred[i]
    return  cls_labels
    
def list_to_colormap(x_list, CLASS_NUM):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([255,182,193]) / 255.
        if item == 2:
            y[index] = np.array([60,179,113]) / 255.
        if item == 3:
            y[index] = np.array([255,165,0]) / 255.
        if item == 4:
            y[index] = np.array([65,105,225]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 6:
            y[index] = np.array([148,0,211]) / 255.
        if item == 7:
            y[index] = np.array([139,69,19]) / 255.
        if item == 8:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 9:
            y[index] = np.array([0,255,255])/255.
        if item == 10:
            y[index] = np.array([128, 128, 0])/255.
        if item == 11:
            y[index] = np.array([255,255,0])/255.
        if item == 12:
            y[index] = np.array([121,255,49])/255.
        if item == 13:
            y[index] = np.array([255,49,183])/255.
        if item == 14:
            y[index] = np.array([112, 192, 188])/255.
        if item == 15:
            y[index] = np.array([183,121,121])/255.
        if item == 16:
            y[index] = np.array([13,0,100]) / 255.
        if item == 17:
            y[index] = np.array([126, 191, 20]) / 255.
        if item == 18:
            y[index] = np.array([116, 89, 120]) / 255.
        if item == 19:
            y[index] = np.array([32, 181, 140]) / 255.
        if item == 20:
            y[index] = np.array([132, 88, 40]) / 255.
        if item == 21:
            y[index] = np.array([76, 49, 240]) / 255.
        if item == CLASS_NUM:
            y[index] = np.array([255,255,255])/255.
    return y
def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def get_cls_map(y_pred, support_label, y, indices, oa, args):
    if args.dataset == 'Yancheng':
        class_num = 18
    elif args.dataset == 'Robinia':
        class_num = 11
    else:
        class_num = 9
    print(y_pred.shape)
    cls_labels = get_classification_map(y_pred+1, y, support_label, indices)
    x = np.ravel(cls_labels)
    gt = y.flatten()
    y_list = list_to_colormap(x,class_num)
    y_gt = list_to_colormap(gt,class_num)
    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))
    classification_map(y_re, y, 300, '/mnt/hdd/yqx/Cross_Domain_FSL_Open_Set_Recognition/classification/' + args.dataset + '_' + str(oa * 100) + '.png')
    classification_map(gt_re, y, 300,'/mnt/hdd/yqx/Cross_Domain_FSL_Open_Set_Recognition/classification/' + args.dataset + '_gt.png')
    print('------Get classification maps successful-------')

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(epoch, opt, optimizer, threshold=1e-6):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0 and opt.learning_rate > threshold:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

