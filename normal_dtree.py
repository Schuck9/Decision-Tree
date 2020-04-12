#! /usr/bin/env python
# -*- coding=utf-8 -*-
# 调用DTree类实现决策树构建（未剪枝）
# 调用Dtreeplot类绘制图形
# Author: USTB-SAEE-Lab701

import initinfo
import copy
import pandas as pd
from dtreeplot import *
from dtree import *



if __name__ == '__main__':
    #数据集读取
    data_train = pd.read_csv('../data/watermelon2-sub2.txt')
    #训练数据与测试数据填充，数据集的第一列为index
    train_data = data_train.values[:,1:].tolist()
    train_remark = data_train.columns.values[1:-1].tolist()
    valid_data = []
    # 数据备份，copy是浅度复制，修改列表里的对象会造成影响
    train_data_f = copy.deepcopy(train_data)
    train_remark_f = copy.deepcopy(train_remark)
    # 决策树生成
    Train = DTree(norm="ent", mode="Normal", choose="last", eps=0.01)
    Train.fit(train_data, valid_data, train_remark)
    # 打印生成的决策树
    print("Tree---->",Train.dtree)
    # 绘制决策树
    fig1 = Dtreeplot(
        figname=Train.mode+"-Trained---using-"+Train.norm)
    fig1.create_plot(Train.dtree)