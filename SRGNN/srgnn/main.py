"""
Created on July, 2018

@author: Tangrizzly
"""


# 参考文献 https://zhuanlan.zhihu.com/p/374403316

import pickle
import time

import numpy as np
from utils import  Data, split_validation
from model import *
import argparse
import os

parser = argparse.ArgumentParser(description='Processing about datasets')
parser.add_argument('--input_path', type=str, default="../DataProcessing/output_Data",help="input file name")
parser.add_argument('--output_path', type=str, default="./output",help="save")
parser.add_argument('--batchSize', type=int, default=100,help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100,help='hidden state size')
parser.add_argument('--epoch', type=int, default=20,help='the number of epochs to train for')
parser.add_argument('--lr', type=int, default= 0.001,help='learning rate')
parser.add_argument('--lr_dc', type=int, default= 0.1,help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default= 3,help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=int, default= 1e-5,help="l2 penalty")
parser.add_argument('--step', type=int, default= 1,help='gnn propogation steps')
parser.add_argument('--patience', type=int, default= 10,help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', type=bool, default= True,help='only use the global preference to predict')
parser.add_argument('--validation', type=bool, default= True,help='validation')
parser.add_argument('--valid_portion', type=int, default=  0.1,help='split the portion of training set as validation set')
parser.add_argument('--dataset', type=str, default="tky")

args = parser.parse_args()

def main(args):
    input_path = os.path.join(args.input_path,args.dataset)
    output_path = os.path(args.output_path,args.dataset)

    if not os.path.exists(output_path):
        os.mkdir(os.path.exists(output_path))

    dataset = args.dataset
    train_data = pickle.load(open(os.path.join(input_path ,'train.txt'), 'rb'))
    test_data = pickle.load(open(os.path.join(input_path , 'test.txt'), 'rb'))
    information = pickle.load(open(os.path.join(input_path , 'information.txt'), 'rb'))
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)  #测试数据调用len(g.node)=309就是下面n_node = 310的来源，为了得到构建GNN的嵌入层的输入节点数目
    train_data = Data(train_data, shuffle=True)
    train_data_length = train_data.inputs.shape[0]
    print(train_data_length)
    # print(train_data)
    test_data = Data(test_data, shuffle=False)
    # print(test_data)
    # del all_train_seq, g
#    information = Data(information,shuffle=False)
    n_node = information[0]
#    n_node = 72
    print("====",information)
    model = trans_to_cuda(SessionGraph(args, n_node))

    # --------------------
    '''
    用于收集Embedding的矩阵
    '''
    list_0 = []
    list_1 = []
    label_list_0 = []
    label_list_1 = []
    # --------------------


    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(args.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr,tem_list,label_list = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            list_0 = tem_list[:]
            label_list_0 = label_list[:]
            print(len(list_0))
            flag = 1

        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            list_1 = tem_list[:]
            label_list_1 = label_list[:]
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= args.patience:
            break

    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))
    print(np.array(list_0).shape)
    print(np.array(label_list_0).shape)
    numpy.save(output_path + "/MMR_BEST_train_" + dataset + "_" + str(args.hiddenSize) + ".npy",np.array(list_0[:train_data_length]))
    numpy.save(output_path + "/ERR_BEST_train_" + dataset + "_" + str(args.hiddenSize) + ".npy",np.array(list_1[:train_data_length]))
    numpy.save(output_path + "/MMR_BEST_test_" + dataset + "_" + str(args.hiddenSize) + ".npy",np.array(list_0[train_data_length:]))
    numpy.save(output_path + "/ERR_BEST_test_" + dataset + "_" + str(args.hiddenSize) + ".npy",np.array(list_1[train_data_length:]))
    numpy.save(output_path + "/MMR_BEST_train_label_" + dataset + "_" + str(args.hiddenSize) + ".npy", np.array(label_list_0[:train_data_length]))
    numpy.save(output_path + "/ERR_BEST_train_label_" + dataset + "_" + str(args.hiddenSize) + ".npy", np.array(label_list_1[:train_data_length]))
    numpy.save(output_path + "/MMR_BEST_test_label_" + dataset + "_" + str(args.hiddenSize) + ".npy", np.array(label_list_0[train_data_length:]))
    numpy.save(output_path + "/ERR_BEST_test_label_" + dataset + "_" + str(args.hiddenSize) + ".npy", np.array(label_list_1[train_data_length:]))

if __name__ == "__main__":
    main(args)