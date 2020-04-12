#! /usr/bin/env python
# -*- coding=utf-8 -*-
# 计算信息熵示例
# 实现数据集watermelon2-sub1信息熵计算
# Author: USTB-SAEE-Lab701

import initinfo
import copy
import pandas as pd
import numpy as np


'''计算数据集信息熵'''
def cal_ent(data):
    """
    @param data: 样本集合（属性值+类别值）
    @return: 数据集信息熵
    """
    ent = 0.0
    label_count = {}
    # 统计各类别样本数
    for sample in data:
        # 样本数据最后一项数据为类别值
        label = sample[-1]
        if label not in label_count.keys():
            label_count[label] = 0
        label_count[label] += 1
    # 计算信息熵
    for key in label_count:
        prob = float(label_count[key]) / len(data)
        ent -= prob * np.log2(prob)
    return ent,label_count


if __name__ == '__main__':
    # 数据集读取
    data_train = pd.read_csv('../data/watermelon2-sub2.txt')
    # data_train = pd.read_csv('../data/watermelon2_valid.txt')
    # 数据填充，数据集的第一列为index，最后一列为label
    train_data = data_train.values[:,1:].tolist()
    # 计算数据集watermelon2-sub1的信息熵
    ent_of_data,label_count = cal_ent(train_data)
    # 显示数据集watermelon2-sub1各类别统计结果与信息熵
    print("labels_of_data:\n",label_count)
    print("ent_of_data:",ent_of_data)