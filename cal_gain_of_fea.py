#! /usr/bin/env python
# -*- coding=utf-8 -*-
# 计算信息增益示例
# 数据集watermelon2-sub2以color属性划分的信息增益计算
# Author: USTB-SAEE-Lab701

import initinfo
import copy
import pandas as pd
import numpy as np


'''计算数据集信息熵'''
def cal_ent(data):
    """
    :param data: 样本集合（属性值+类别值）
    :return: 数据集信息熵
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


'''划分数据集（离散）'''
def split_data(data, axis, value):
    """
    :param data: 样本集合（属性值+类别值）
    :param axis: 划分属性的索引
    :param value: 划分属性取值
    :return: 以划分属性进行划分后的数据集
    """
    retdata = []
    for sample in data:
        resample = copy.deepcopy(sample)
        if sample[axis] == value:
            retdata.append(resample)
        
    return retdata


'''计算数据集某一属性的信息增益'''
def cal_gain_fea(data,axis):
    """
    :param data: 样本集合（属性值+类别值）
    :param axis: 划分属性的索引
    :return: 数据集信息熵
    """
    # 获取数据集某一属性下的数据
    feavallist = [sample[axis] for sample in data]
    # 获取该属性所有的属性值（不重复）
    values = set(feavallist)
    # 对每个属性值均生成数据子集，计算信息熵
    gain = 0.0        #信息增益
    new_ent = 0.0     #各子集的加权信息熵
    subdata_ents = {} #各子集的信息熵记录
    subdata_labels = {} #各子集的类别统计记录
    base_ent,label_count = cal_ent(data)
    for value in values:
        # 根据不同属性值划分数据集
        subdata = split_data(data, axis, value)
        subdata_ent,sublabel_count = cal_ent(subdata)
        prob = len(subdata)/float(len(data))
        new_ent += prob*subdata_ent        
        # 记录各子集的情况（信息熵，各类别的统计情况）
        subdata_ents[value] = subdata_ent
        subdata_labels[value] = sublabel_count
    gain = base_ent - new_ent

    return gain,subdata_ents,subdata_labels



if __name__ == '__main__':
    # 数据集读取
    data_train = pd.read_csv('../data/watermelon2-sub2.txt')
    # data_train = pd.read_csv('../数据集/watermelon2-sub2.txt')
    # 数据填充，数据集的第一列为index，最后一列为label
    train_data = data_train.values[:,1:].tolist()
    train_remark = data_train.columns.values[1:-1].tolist()
    # 计算数据集watermelon2-sub2的信息熵
    base_ent,label_count = cal_ent(train_data)
    # 显示数据集watermelon2-sub2的各类别统计结果与信息熵
    print("labels_of_data:",label_count)
    print("ent_of_data:",base_ent,"\n")
    # 计算数据集某一属性信息增益(第一列为color,最后一列为label)
    for fea_index in range(len(train_data[0])-1):
        gain, subdata_ents, subdata_labels = \
                            cal_gain_fea(train_data,fea_index)
        # 输出各子集的统计结果与信息熵
        for subdata,sublabel in subdata_labels.items():
            print("feature_vaule:",subdata)
            print("label_count:",sublabel,
                "  ent_of_subdata:",subdata_ents[subdata])
        # 输出以color属性进行数据集划分的信息增益
        print("Gain(D,%s)=%f\n" % (train_remark[fea_index],gain))