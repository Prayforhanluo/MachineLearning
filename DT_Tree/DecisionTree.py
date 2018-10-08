# -*- coding: utf-8 -*-
# @Time  : 2018/10/7 11:15
# @Author : han luo
# @git   : https://github.com/Prayforhanluo

"""
    决策树是一类用来分类和回归的无参监督学习方法。其目的是创建一种模型从数据特征中学习简单的决策规则来预测一个
    目标变量的值。通俗的说决策树就像是一个带有终止模块的流程图。决策树的算法：

    ID3 ： 使用信息增益大小来判断当前节点应该用什么特征老构建决策树，用计算出的信息增益最大的特征来建立决策树
           的当前节点。

    C4.5： 使用信息增益比(信息增益和特征熵的比值)来作为属性特征的选择标准，在决策树构造的同时进行剪枝操作；避免
           树的过度拟合。

    CART： CART分类树算法使用基尼系数来代替信息增益比，基尼系数代表了模型的不纯度，基尼系数越小，则不纯度越低，特征越好。
"""

from __future__ import division
from math import log
import copy
import operator
import matplotlib.pyplot as plt


def ShannonEnt(dataSet):
    """
        计算信息熵。
    :param dataSet:
    :return:
    """
    # 特征出现次数统计
    numEntries = len(dataSet)
    labelCounts = {}
    for feaVec in dataSet:
        currentLabel = feaVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0
    # 计算香农熵
    for key, count in labelCounts.items():
        prob = float(count) / numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt


def splitDataSet(dataSet,axis,value):
    """
    按照给定特征划分数据集(针对离散型属性)
    :param dataSet: 待划分的数据集
    :param axis:  划分数据的特征
    :param value:   特征值
    :return: 划分后的数据集
    """
    retDataSet = []
    for featVec in dataSet:
        # 符合特征的值
        if featVec[axis] == value:
            # 符合特征->存储，不需要存储选为划分的特征
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def splitSerialDataSet(dataSet, axis, value, direction):
    """
    与离散的划分类似，区别在于处理连续特征值对连续变量划分数据集
    决定是划分小于value的样本还是大于value的样本

    :param dataSet: 待划分的数据集
    :param axis: 划分数据的特征
    :param value: 特征值
    :param direction: 划分的方向
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if direction == 0:
            if featVec[axis] > value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
        else:
            if featVec[axis] <= value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet, labels):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = ShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    bestSplitDict = {}
    for i in range(numFeatures):
        # 对连续型特征进行处理 ,i代表第i个特征,featList是每次选取一个特征之后这个特征的所有样本对应的数据
        featList = [example[i] for example in dataSet]
        # 因为特征分为连续值和离散值特征，对这两种特征需要分开进行处理。
        if isinstance(featList[0],float) == True or isinstance(featList[0],int) == True:
            # 产生n-1个候选划分点
            sortfeatList = sorted(featList)
            splitList = []
            for j in range(len(sortfeatList) - 1):
                splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2.0)
            bestSplitEntropy = 10000
            # 求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点  找到最大信息熵的划分
            for value in splitList:
                newEntropy = 0.0
                # 根据value将属性集分为两个部分
                subDataSet0 = splitSerialDataSet(dataSet, i, value, 0)
                subDataSet1 = splitSerialDataSet(dataSet, i, value, 1)

                prob0 = len(subDataSet0) / float(len(dataSet))
                newEntropy += prob0 * ShannonEnt(subDataSet0)
                prob1 = len(subDataSet1) / float(len(dataSet))
                newEntropy += prob1 * ShannonEnt(subDataSet1)
                if newEntropy < bestSplitEntropy:
                    bestSplitEntropy = newEntropy
                    bestSplit = value
            # 用字典记录当前特征的最佳划分点
            bestSplitDict[labels[i]] = bestSplit
            infoGain = baseEntropy - bestSplitEntropy

        # 对离散型特征进行处理
        else:
            uniqueVals = set(featList)
            newEntropy = 0.0
            # 计算该特征下每种划分的信息熵,选取第i个特征的值为value的子集
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * ShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    # 若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理
    # 即是否小于等于bestSplitValue,例如将密度变为密度<=0.3815
    # 将属性变了之后，之前的那些float型的值也要相应变为0和1
    if type(dataSet[0][bestFeature]).__name__=='float' or type(dataSet[0][bestFeature]).__name__ == 'int':
        bestSplitValue = bestSplitDict[labels[bestFeature]]
        labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)
        for i in range(len(dataSet)):
            if dataSet[i][bestFeature] <= bestSplitValue:
                dataSet[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature] = 0
    return bestFeature


def majorityCount(classList):
    """
    计算一个特征数据列表中 出现次数最多的特征值以及次数
    :param classList: 特征值列表
    :return: 次数最多的特征值
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def createdTree(dataSet, labels, data_all, labels_all):
    """
        递归产生决策树，终止条件：
        1.当前节点包含的样本全部属于同一类别
        2.当前属性集为空，即所有可以用来划分的属性全部用完了
        3.当前节点所包含的样本集合为空
    :param dataSet: 用于构建树的数据集
    :param labels:  剩下的用于划分的类别
    :param data_all:   全部数据集
    :param labels_all:  全部类别
    :return:
    """
    labels_all_copy = copy.deepcopy(labels_all)
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        # 当前节点都是同一类别
        return classList[0]
    if len(dataSet[0]) == 1:
        # 当前属性为空
        return majorityCount(classList)

    # bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeat = chooseBestFeatureToSplit(dataSet,labels)
    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel:{}}
    featValues = [example[bestFeat] for example in dataSet]

    uniqueVals = set(featValues)
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        currentlabel = labels_all_copy.index(bestFeatLabel)
        featValuesFull=[example[currentlabel] for example in data_all]
        uniqueValsFull = set(featValuesFull)
    del (labels[bestFeat])
    for value in uniqueVals:
        subLabels = labels[:]
        if type(dataSet[0][bestFeat]).__name__ == 'str':
            uniqueValsFull.remove(value)
        myTree[bestFeatLabel][value] = createdTree(splitDataSet(dataSet, bestFeat, value), subLabels, data_all, labels_all_copy)
    # 完成对缺失值的处理
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        for value in uniqueValsFull:
            myTree[bestFeatLabel][value] = majorityCount(classList)
    return myTree


def classify(inputTree,featLabels,testVec):
    """
    对未知特征在创建的决策树上分类
    :param inputTree: 决策树
    :param featLabels: 特征
    :param testVec:
    :return:  识别结果
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key :
            if isinstance(secondDict[key],dict):
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


"""
    决策树的可视化
"""
decisionNode = dict(boxstyle = "sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
plt.rcParams['font.sans-serif']=['SimHei']


def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',\
    xytext=centerPt,textcoords='axes fraction',\
    va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]# 获得数据字典中键值列表 并返回第一个值
    secondDict = myTree[firstStr]# 获取第一个键值的值
    for key in secondDict.keys():
        if isinstance(secondDict[key],dict):# 判断数据类型
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth :
            maxDepth = thisDepth
    return maxDepth


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key],dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs +=1
    return numLeafs


def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)


def plotTree(myTree,parentPt,nodeTxt):
    numLeafs = getNumLeafs(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if isinstance(secondDict[key],dict):
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff +1.0/plotTree.totalD


def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])

    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops) # 绘制子图
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()


if __name__ == '__main__':
    f = open('DT_data.txt')
    xigua = [i.strip().split() for i in f.readlines()]
    xigua = [[float(i) if '.' in i else i for i in row] for row in xigua]
    Labels = ['color', 'root', 'knock', 'texture', 'navel', 'touch', 'density', 'sugar']
    xiguaTree = createdTree(xigua,Labels,xigua,Labels)
    createPlot(xiguaTree)