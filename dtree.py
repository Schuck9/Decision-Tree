#! /usr/bin/env python
# -*- coding=utf-8 -*-
# 决策树类，包含ID3以及C4.5算法，但略有出入
# 以信息增益(ent)、信息增益率(gr)作为准则
# 包含连续值处理以及剪枝处理
# Author: USTB-SAEE-Lab701
from numpy import *
import numpy as np
import copy
import pickle


class DTree(object):

    def __init__(self, eps=0.01, norm="ent", mode="Normal", choose="first", name=None):
        self.dtree = dict() #决策树
        self.eps = eps      #节点期望收益
        self.norm = norm    #准则; ent: 信息增益，gr: 信息增益率
        self.mode = mode    #模式; Normal: 正常生长, Pre: 预剪枝
        self.choose = choose  #出现多个最优时，选择倾向; fisrt或last
        if not name:
            self.name = "Decision Tree"
        else:
            self.name = name

    '''数据填充（训练）'''
    def fit(self, data_train, data_valid, remark):
        """
        :param data_train: 训练样本集合（属性值+类别值）
        :param data_valid: 验证样本集合（属性值+类别值）
        :param remark: 候选属性集合（remark[0]: 名称; remark[1]: 连续性）
        :return: 构建完毕的决策树，以基尼指数(gini)作为准则时默认构建CART
                 以信息增益(ent)、信息增益率(gr)作为准则时默认构建决策树
        """
        assert (data_train and remark),"未输入训练数据或候选属性集合"
        # 候选属性集合加入候选属性连续性标签（remark[0]: 名称; remark[1]: 连续性）
        remark = [remark, [is_continuous(fea) for fea in data_train[0][:-1]]]
        if (self.norm == "ent" or self.norm == "gr"):
            self.dtree = self._build_tree(data_train, data_valid, remark)
        else:
            print("仅支持信息增益(ent)与信息增益率(gr)")
        return self.dtree


    '''生成决策树'''
    def _build_tree(self, data_train, data_valid, remark):
        """
        :param data_train: 训练样本集合（属性值+类别值）
        :param data_valid: 验证样本集合（属性值+类别值）
        :param remark: 候选属性集合（remark[0]: 名称; remark[1]: 连续性）
        :return: dict型的树结构（键对应划分属性，键值对应以该属性划分后的子树）
        """
        # 获取类别标签
        label = [sample[-1] for sample in data_train]
        # 若所有样本都属于同一个类别，则以该类别标记当前节点
        if label.count(label[0]) == len(label):
            return label[0]
        # 若候选属性集合为空，则将样本数目最多的类视为当前节点的类别
        if len(data_train[0]) == 1:
            return self._vote_class(data_train)
        # 在候选属性集合以某一准则选择最优属性（信息增益、信息增益率）
        best_fea, best_val, best_gain = self._choose_bestfea(data_train, remark)
        # 若无法选出最优划分属性或收益低于预期，不继续生长，根据样本类别投票
        if (best_fea == -1) or (best_gain < self.eps):
            return self._vote_class(data_train)
        # 若进行预剪枝，计算划分前后的准确率，判定是否进一步生长
        if self.mode=="Pre" and data_valid:
            preprun = self._judge_preprun(data_valid, best_fea, best_val, remark)
            if preprun:
                return self._vote_class(data_train)
        # 数据复制，避免影响原始数据复用
        train_temp = copy.deepcopy(data_train)
        valid_temp = copy.deepcopy(data_valid)
        remark_temp = copy.deepcopy(remark)
        # 若最佳划分属性为连续属性值，则以划分点为界进行二值化处理
        if remark_temp[1][best_fea]:
            remark_temp[0][best_fea] = remark_temp[0][best_fea]+'<='+str(round(best_val,5))
            # 训练集处理
            for i in range(len(train_temp)):
                train_temp[i][best_fea] = "Yes" if train_temp[i][best_fea] <= best_val else "No"
            # 验证集处理
            for i in range(len(valid_temp)):
                valid_temp[i][best_fea] = "Yes" if valid_temp[i][best_fea] <= best_val else "No"
        # 记录根节点信息
        b_remark = remark_temp[0][best_fea]
        Tree = {b_remark: {}}
        # 将已选中属性从候选属性集合中移除
        for i in range(len(remark_temp)):
            del(remark_temp[i][best_fea])
        # 针对bestfea的每个取值，划分数据集并生成子树
        feavalset = set([sample[best_fea] for sample in train_temp])
        for value in feavalset:
            sub_data_t = self._split_data(train_temp, best_fea, value, False, 0, True)
            sub_data_v = self._split_data(valid_temp, best_fea, value, False, 0, True)
            Tree[b_remark][value] = self._build_tree(sub_data_t, sub_data_v, remark_temp)

        return Tree

    '''选择最佳属性划分数据集'''
    def _choose_bestfea(self, data, remark):
        """
        :param data: 样本集合（属性值+类别值）
        :param remark: 候选属性集合（remark[0]: 名称; remark[1]: 连续性）
        :return: 最佳划分属性索引, 最佳划分点以及最佳划分的收益
        """
        best_fea = -1
        best_val = -1
        best_gain = 0.0        
        best_split = {}
        #计算数据集信息熵
        base_ent = cal_ent(data)
        # 对候选属性集合中的每个属性分别计算收益
        for fea in range(len(data[0])-1):
            continuous = remark[1][fea]
            feavallist = [sample[fea] for sample in data]

            # 对连续属性值进行处理
            if continuous:
                # 产生n-1个候选划分点
                # print(feavallist)
                sortfeaval = sorted(feavallist)
                splitlist = [(sortfeaval[i]+sortfeaval[i+1])/2.0 
                                for i in range(len(sortfeaval)-1)]

                # 求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点

                temp_info = 1
                for point,splitvalue in enumerate(splitlist):
                	# 计算不同准则下划分的信息收益
                    sub_1 = []
                    sub_2 = []	
                    for element in data:
                        if element[fea] <= splitvalue:
                            sub_1.append(element)
                        else :
                            sub_2.append(element)

                    new_info = len(sub_1)/len(sortfeaval)*cal_ent(sub_1)+len(sub_2)/len(sortfeaval)*cal_ent(sub_2)
                    value = splitvalue
                
                    # 更新收益
                    if new_info <  temp_info:
                        temp_info = new_info
                        best_point = point
                # 记录当前特征的最佳划分点
                best_split[remark[0][fea]] = splitlist[best_point]
                current_info = temp_info
            # 对离散属性值进行处理
            else:
                values = set(feavallist)
                new_ent = 0.0
                # 计算该属性下每种划分的信息熵
                for value in values:
                    subdata = self._split_data(data, fea, value, continuous, 0)
                    prob = len(subdata)/float(len(data))
                    new_ent += prob*cal_ent(subdata)
                # 不同准则信息收益计算
                current_info = base_ent - new_ent
                if self.norm != "gr": 
                    current_info = base_ent - new_ent
                    # print(feavallist)
                else:
                    current_info = base_ent - new_ent
                    feavallist = [[sample] for sample in feavallist]
                    fea_ent = cal_ent(feavallist)
                    current_info = (base_ent - new_ent)/fea_ent

            if self.choose == "first":
                if current_info > best_gain:
                    best_gain = current_info
                    best_fea = fea
            elif self.choose == "last":
                if current_info >= best_gain:
                    best_gain = current_info
                    best_fea = fea
        # 若最佳划分属性为连续属性值，则记录之前的划分点
        if remark[1][best_fea]:
            best_val = best_split[remark[0][best_fea]]

        return best_fea,best_val,best_gain


    '''决策树预测'''
    def predict(self, testdata, remark, inTree=None):
        """
        :param testdata: 测试样本集合（属性值+类别值）
        :param remark: 候选属性集合
        :param inTree: dict型的树结构
        :return: 测试样本的预测类别向量
        """
        # 若输入的为整个测试样本集合，则进行拆分
        if (is_list(testdata) and is_list(testdata[0])):
            rst = []
            for sub in range(len(testdata)):
                rst.append(self.predict(testdata[sub], remark, inTree))
            return rst

        if not inTree:
            inTree = self.tree
        # 对于每一个样本进行预测
        root_node = list(inTree.keys())[0]  # 根节点
        root_remark = root_node
        # 如果是连续型的属性，取"<="之前的部分作为节点值
        less_index = str(root_node).find('<')
        if less_index > -1:
            root_remark = str(root_node)[:less_index]
        secondary = inTree[root_node]
        fea_index = remark.index(root_remark)
        classlabel = None
        secondary_temp = copy.deepcopy(secondary)
        for key in secondary_temp.keys():
            # 处理连续属性值
            if is_continuous(testdata[fea_index]):
                pointval = float(str(root_node)[less_index + 2:])
                # 进入左右子树
                if testdata[fea_index] <= pointval:
                    # 若分支不是叶节点，递归，直至叶节点返回类别标签
                    if is_dict(secondary['Yes']):
                        classlabel = self.predict(testdata,remark,secondary['Yes'])
                    else:
                        return secondary['Yes']
                else:
                    if is_dict(secondary['No']):
                        classlabel = self.predict(testdata,remark,secondary['No'])
                    else:
                        return secondary['No']
            else:
                if testdata[fea_index] == key:
                    # 若分支不是叶节点，递归，直至叶节点返回类别标签
                    if is_dict(secondary[key]):
                        classlabel = self.predict(testdata,remark,secondary[key])
                    else:
                        return secondary[key]

        return classlabel

    '''判断是否需要进行预剪枝'''
    def _judge_preprun(self, data_valid, best_fea, best_val, remark):
        """
        :param data_valid: 验证样本集合（属性值+类别值）
        :param best_fea: 划分属性
        :param best_val: 划分属性值
        :param remark: 候选属性集合（remark[0]: 名称; remark[1]: 连续性）
        :return: 是否需要进行预剪枝
        """
        preprun = False
        acc_after = 0.0
        acc_pre = self._testing_major(data_valid, self._vote_class(data_valid))
        # 
        if acc_pre>((len(data_valid)-1)/len(data_valid)) or not data_valid:
            return preprun
        # 若属性值连续
        if remark[1][best_fea]:
            for i in range(2):
                sub_data = self._split_data(data_valid, best_fea, best_val, True, i)
                acc_after += len(sub_data)*self._testing_major(
                                    sub_data, self._vote_class(sub_data))
        else: 
            feavalset = set([sample[best_fea] for sample in data_valid])
            for value in feavalset:
                sub_data = self._split_data(data_valid, best_fea, value, False, 0)
                acc_after += len(sub_data)*self._testing_major(
                                    sub_data, self._vote_class(sub_data))

        acc_after /= len(data_valid)
        # 若不能增加准确率，则不进行生长
        if acc_after <= acc_pre:
            preprun = True

        return preprun
        
    '''决策树后剪枝(REP)'''
    def post_pruning(self, data_train, data_valid, remark, inTree=None):
        """
        :param data_train: 训练样本集合（属性值+类别值）
        :param data_valid: 验证样本集合（属性值+类别值）
        :param remark: 候选属性集合
        :param inTree: 构建完毕的决策树
        :return: 后剪枝后的决策树
        """
        if not inTree:
            inTree = self.tree
        # 若验证集在该处无数据，则不进行剪枝
        if not data_valid:
            return inTree
        root_node = list(inTree.keys())[0]
        root_remark = root_node
        less_index = str(root_node).find('<')
        # 如果是连续型的特征
        if less_index > -1:
            root_remark = str(root_node)[:less_index]
            pointval = float(str(root_node)[less_index + 2:])
        secondary = inTree[root_node]
        fea_index = remark.index(root_remark)
        temp_remarks = copy.deepcopy(remark)
        # 对每个分支
        for key in secondary.keys():
            # 若不为叶节点，进一步分裂数据集并剪枝
            if is_dict(secondary[key]): 	
                if is_continuous(data_train[0][fea_index]):
                    if key == 'Yes':
                        subdata_train = self._split_data(data_train, fea_index, pointval, True, 0, False)
                        subdata_valid = self._split_data(data_valid, fea_index, pointval, True, 0, False)
                    elif key == 'No':
                        subdata_train = self._split_data(data_train, fea_index, pointval, True, 1, False)
                        subdata_valid = self._split_data(data_valid, fea_index, pointval, True, 1, False)
                else:
                    subdata_train = self._split_data(data_train, fea_index, key, False, 0, False)
                    subdata_valid = self._split_data(data_valid, fea_index, key, False, 0, False)
                # 分裂数据集并剪枝
                inTree[root_node][key] = self.post_pruning(
                    subdata_train, subdata_valid, temp_remarks, secondary[key])
        # 若剪枝不能提升泛化性能，则不进行剪枝
        if (self._testing_dtree(data_valid, temp_remarks, inTree) >
            self._testing_major(data_valid, self._vote_class(data_train))):
            return inTree
        return self._vote_class(data_train)


    '''划分数据集'''
    def _split_data(self, data, axis, value, v_c=False, direct=0, remove=False):
        """
        :param data: 样本集合（属性值+类别值）
        :param axis: 划分属性的索引
        :param value: 划分属性取值
        :param v_c: 属性值是否连续，连续属性值v_c：true
        :param direct: 划分的方向(小于等于：0，大于：1)
        :param remove: 划分数据集时是否移除索引对应部分
        :return: 以划分属性进行划分后的数据集
        """
        retdata = []
        for sample in data:
            resample = copy.deepcopy(sample)
            if remove:
                resample.pop(axis)
            if not v_c:
                if sample[axis] == value:
                    retdata.append(resample)
            else:
                if direct == 0:
                    if float(sample[axis]) <= value:
                        retdata.append(resample)
                elif direct == 1:
                    if float(sample[axis]) > value:
                        retdata.append(resample)              
        return retdata
    
    '''根据样本类别进行投票'''
    def _vote_class(self, data):
        """
        :param data: 进行投票的样本集
        :return: 类别标签向量中出现频率最高的类别
        """
        vote = {}
        labels = [sample[-1] for sample in data]
        for label in labels:
            if label not in vote.keys():
                vote[label] = 0
            vote[label] += 1
        return list(vote.keys())[list(vote.values()).index(max(vote.values()))]

    '''测试验证集准确率（使用决策树）'''
    def _testing_dtree(self, data, remark, inTree):
        """
        :param data: 训练样本集合（属性值+类别值）
        :param remark: 候选属性集合
        :param inTree: 输入的决策树
        :return: 验证集准确率
        """
        if (not data) or (not remark):
            return 0.0
        accuracy=0.0
        for i in range(len(data)):
            if self.predict(data[i], remark, inTree)==data[i][-1]:
                accuracy+=1
        return float(accuracy/len(data))

    '''测试数据集准确率(使用投票)'''
    def _testing_major(self, data, label):
        """
        :param data: 训练样本集合（属性值+类别值）
        :param label: 预测的类别值
        :return: 使用投票得到的验证集准确率
        """
        if (not data) or (not label):
            return 0.0
        accuracy=0.0
        for i in range(len(data)):
            if data[i][-1]==label:
                accuracy+=1
        return float(accuracy/len(data))


    # 存储决策树
    def store_tree(self, inTree, filename):      
        fw = open(filename, 'w')
        pickle.dump(inTree, fw)
        fw.close()
 
    # 读取决策树, 文件不存在返回None
    def load_tree(self, filename):
        if os.path.isfile(filename):
            treeflie = open(filename)
            return pickle.load(treeflie)
        else:
            return None


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
    return ent


'''检查是否连续'''
def is_continuous(x):
    return type(x).__name__ == 'float' or type(x).__name__ == 'int'

'''检查是否为...类型'''
def is_string(x):
    return type(x).__name__ == 'str'
def is_dict(x):
    return type(x).__name__ == 'dict'
def is_list(x):
    return type(x).__name__ == 'list'
