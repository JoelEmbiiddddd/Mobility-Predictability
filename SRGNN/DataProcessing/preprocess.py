#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import argparse

def main(args):

    # 设置输出路径、输入文件路径、类别数、总列表和数据集
    output_path = os.path.join("output_Data", args.output_path)
    input_file = os.path.join("data", args.input_file)
    classes = 0
    total_list = []
    dataset = args.dataset
    print("-- Starting @ %ss" % datetime.datetime.now())

    # 读取输入文件，将数据按session_id、item_id和时间戳进行分组，并按时间戳排序。
    with open(input_file, "r") as f:
        reader = csv.DictReader(f, delimiter=',')
        sess_clicks = {}
        sess_date = {}
        ctr = 0
        curid = -1
        curdate = None
        for data in reader:
            sessid = data['session_id']
            if curdate and not curid == sessid:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
                sess_date[curid] = date
            curid = sessid
            item = data['item_id'], int(data['timeframe'])
            curdate = data['eventdate']

            if sessid in sess_clicks:
                sess_clicks[sessid] += [item]
            else:
                sess_clicks[sessid] = [item]
            ctr += 1

        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]
        sess_date[curid] = date
    print("-- Reading data @ %ss" % datetime.datetime.now())

    # 过滤掉只有一个item的session
    for s in list(sess_clicks):
        if len(sess_clicks[s]) == 1:
            del sess_clicks[s]
            del sess_date[s]

    # 计算每个item出现的次数，并按次数排序
    iid_counts = {}
    for s in sess_clicks:
        seq = sess_clicks[s]
        for iid in seq:
            if iid in iid_counts:
                iid_counts[iid] += 1
            else:
                iid_counts[iid] = 1

    sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))


    # 过滤掉出现次数小于2次的item，并将结果保存在sess_clicks中
    length = len(sess_clicks)
    for s in list(sess_clicks):
        curseq = sess_clicks[s]
        filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
        if len(filseq) < 2:
            del sess_clicks[s]
            del sess_date[s]
        else:
            sess_clicks[s] = filseq

    # Split out test set based on dates
    dates = list(sess_date.items())
    maxdate = dates[0][1]

    for _, date in dates:
        if maxdate < date:
            maxdate = date

    splitdate = maxdate - 86400 * 7

    print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
    tra_sess = filter(lambda x: x[1] < splitdate, dates)
    tes_sess = filter(lambda x: x[1] > splitdate, dates)

    # 将数据集按时间戳划分为训练集和测试集，并按时间戳排序
    tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
    tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
    print(len(tra_sess))    # 186670    # 7966257
    print(len(tes_sess))    # 15979     # 15324
    print(tra_sess[:3])
    print(tes_sess[:3])
    print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

    # Choosing item count >=5 gives approximately the same number of items as reported in paper
    item_dict = {}
    # Convert training sessions to sequences and renumber items to start from 1

    # 将测试集中的item转换为数字，并返回测试集的id、日期和序列
    def obtian_tra():
        train_ids = []
        train_seqs = []
        train_dates = []
        item_ctr = 1
        for s, date in tra_sess:
            seq = sess_clicks[s]
            outseq = []
            for i in seq:
                if i in item_dict:
                    outseq += [item_dict[i]]
                else:
                    outseq += [item_ctr]
                    item_dict[i] = item_ctr
                    item_ctr += 1
            if len(outseq) < 2:  # Doesn't occur
                continue
            train_ids += [s]
            train_dates += [date]
            train_seqs += [outseq]
        print("classes=======",item_ctr)     # 43098, 37484
        classes = item_ctr
        total_list.append(classes)
        return train_ids, train_dates, train_seqs


    # Convert test sessions to sequences, ignoring items that do not appear in training set
    def obtian_tes():
        test_ids = []
        test_seqs = []
        test_dates = []
        for s, date in tes_sess:
            seq = sess_clicks[s]
            outseq = []
            for i in seq:
                if i in item_dict:
                    outseq += [item_dict[i]]
            if len(outseq) < 2:
                continue
            test_ids += [s]
            test_dates += [date]
            test_seqs += [outseq]
        return test_ids, test_dates, test_seqs


    tra_ids, tra_dates, tra_seqs = obtian_tra()
    tes_ids, tes_dates, tes_seqs = obtian_tes()


    def process_seqs(iseqs, idates):
        out_seqs = []
        out_dates = []
        labs = []
        ids = []
        for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
            for i in range(1, len(seq)):
                tar = seq[-i]
                labs += [tar]
                out_seqs += [seq[:-i]]
                out_dates += [date]
                ids += [id]
        return out_seqs, out_dates, labs, ids


    tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
    te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
    tra = (tr_seqs, tr_labs)
    tes = (te_seqs, te_labs)
    print(len(tr_seqs))
    print(len(te_seqs))
    total_list.append(len(tr_seqs))
    total_list.append(len(te_seqs))
    print(total_list)
    print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
    print(te_seqs[:3], te_dates[:3], te_labs[:3])
    all = 0

    for seq in tra_seqs:
        all += len(seq)
    for seq in tes_seqs:
        all += len(seq)
    print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))
    print(len(tra_seqs))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pickle.dump(tra, open(output_path + "/train.txt", 'wb'))
    pickle.dump(tes, open(output_path + "/test.txt", 'wb'))
    pickle.dump(tra_seqs, open(output_path + "/all_train_seq.txt", 'wb'))
    pickle.dump(total_list,open(output_path + "/information.txt",'wb'))
    print('Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing about datasets')
    parser.add_argument('--output_path', type=str, default="tky")
    parser.add_argument('--input_file', type=str, default="tky.csv")
    parser.add_argument('--dataset', type=str, default="tky")

    args = parser.parse_args()
    main(args)