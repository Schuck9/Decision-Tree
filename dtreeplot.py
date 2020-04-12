#! /usr/bin/env python
# -*- coding=utf-8 -*-
# 决策树绘制类，绘制决策树示意图
# Author: USTB-SAEE-Lab701
import matplotlib.pyplot as plt

# 节点绘制参数设置
D_NODE = dict(boxstyle="sawtooth", fc="0.9")
L_NODE = dict(boxstyle="round4", fc="0.9")
ARROW = dict(arrowstyle="<-")


class Dtreeplot(object):

    def __init__(self, figname=None):
        self.ax1_ = None
        self.x0ff_ = 0.0
        self.y0ff_ = 0.0
        self.totalw_ = 0.0
        self.totald_ = 0.0
        if not figname:
            self.figname = "Fig DTree"
        else:
            self.figname = figname

    # 计算树的叶子节点数量
    def _get_leafnum(self, myTree):
        leafnums = 0
        secondary = myTree[list(myTree.keys())[0]]
        for key in secondary.keys():
            if type(secondary[key]).__name__ == 'dict':
                leafnums += self._get_leafnum(secondary[key])
            else:
                leafnums += 1
        return leafnums
    
    # 计算树的最大深度
    def _get_treedepth(self, myTree):
        max_depth = 0
        secondary = myTree[list(myTree.keys())[0]]
        for key in secondary.keys():
            if type(secondary[key]).__name__ == 'dict':
                this_depth = 1+self._get_treedepth(secondary[key])
            else:
                this_depth = 1
            if this_depth > max_depth:
                max_depth = this_depth
        return max_depth

    # 创建绘画节点
    def create_plot(self, inTree):
        # 若输入的树根节点即为叶节点或为空，则提示错误
        assert (type(inTree).__name__ == 'dict' and inTree),\
            "Root node shouldn't be leaf node!"
        fig = plt.figure(self.figname, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.ax1_ = plt.subplot(111, frameon=False, **axprops)
        self.totalw_ = float(self._get_leafnum(inTree))
        self.totald_ = float(self._get_treedepth(inTree))
        self.x0ff_ = -0.5/self.totalw_
        self.y0ff_ = 1.0
        self._plot_tree(inTree, (0.5, 1.0), '')
        plt.show()
 
    # 绘制节点
    def _plot_node(self, nodetxt, centerpt, parentpt, nodetype):
        self.ax1_.annotate(nodetxt,
            xy=parentpt, xycoords='axes fraction', xytext=centerpt, 
            textcoords='axes fraction', va="center", ha="center", 
            bbox=nodetype, arrowprops=ARROW)

    # 绘制箭头文本
    def _plot_midtext(self, cntrpt, parentpt, txtstring):
        lens = len(txtstring)
        mid_x = (parentpt[0]+cntrpt[0])/2.0-lens*0.002
        mid_y = (parentpt[1]+cntrpt[1])/2.0
        self.ax1_.text(mid_x, mid_y, txtstring)

    # 绘制决策树
    def _plot_tree(self, myTree, parentpt, nodetxt):
        leafnums = self._get_leafnum(myTree)
        depth = self._get_treedepth(myTree)
        root_node = list(myTree.keys())[0]
        cntrpt = (self.x0ff_+(1.0+float(leafnums)) /
                  2.0/self.totalw_, self.y0ff_)
        self._plot_midtext(cntrpt, parentpt, nodetxt)
        self._plot_node(root_node, cntrpt, parentpt, D_NODE)
        secondary = myTree[root_node]
        self.y0ff_ = self.y0ff_-1.0/self.totald_
        for key in secondary.keys():
            if type(secondary[key]).__name__ == 'dict':
                self._plot_tree(secondary[key], cntrpt, str(key))
            else:
                self.x0ff_ = self.x0ff_+1.0/self.totalw_
                self._plot_node(secondary[key], 
                    (self.x0ff_, self.y0ff_), cntrpt, L_NODE)
                self._plot_midtext((self.x0ff_, self.y0ff_), cntrpt, str(key))
        self.y0ff_ = self.y0ff_+1.0/self.totald_