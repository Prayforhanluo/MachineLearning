# -*- coding: utf-8 -*-
# @Time  : 2018/10/7 11:15
# @Author : han luo
# @git   : https://github.com/Prayforhanluo

from __future__ import division
import numpy as np
import pandas as pd
import operator
from sklearn.neighbors import KNeighborsClassifier
from itertools import islice
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

"""
    利用sklearn 上面的示例数据，以及sklearn包里面的KNN的算法来测试。
    KNN 算法小结(自定义)：
        1. 将已知类别的数据准备好。
        2.对于一个给与的未知类别的数据，计算它与已知类别数据的每一个点的距离。
        3.按照距离排序。
        4.确定前K个点所在的类别的出现频率，出现最多的类即是这个未知数据的类别。
    KNN 算法是一种有监督的算法。
    注意：K参数并非越大越好。
"""


def DataPrepare():
    """
    Loading the breast cancer data.
    :return:
    cancer DataFrame
    """
    cancer_data = load_breast_cancer()
    data_feature = cancer_data['feature_names']

    df = pd.DataFrame(data = cancer_data['data'], index = range(0,len(cancer_data['data'])), columns=data_feature)
    df['target'] = cancer_data['target']

    return df


def TargetDistribution():
    """
    calculate the target distribution of data
    :return:
    distribution of data
    """
    cancer_data = DataPrepare()

    class_0 = cancer_data[cancer_data['target'] == 0].sum() # malignant sample count
    class_1 = cancer_data[cancer_data['target'] == 1].sum() # benign sample count

    distribution = pd.Series([class_0,class_1],index=['malignant','benign'])

    return distribution


def DataSplit():
    """
    Split data into train and test.
    :return:
    """
    cancer_data = DataPrepare()

    x = cancer_data[cancer_data.columns]
    y = cancer_data['target']

    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0)
    return x_train, x_test, y_train, y_test


def KNNClassifier(K):
    """

    KNN Classifier by Sklearn.
    Train the mode.
    :param K:  the K point.
    :return:
    """
    x_train, x_test, y_train, y_test = DataSplit()
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(x_train,y_train)

    return knn


def KNNTest(K=1):
    """
    KNN Test
    :param K:  the k point
    :return: the score of KNN model
    """
    x_train, x_test, y_train, y_test = DataSplit()
    knn = KNNClassifier(K)
    score = knn.score(x_test,y_test)

    return score


"""
    Machine Learning in Action 实现方式。
"""


def SKLearnData2File(datafil):
    """
        将 Sklearn 上面的数据转化为最常见的文本格式。
        并以此文本格式文件为起始。
    :param datafil: Out data file. text format.
    :return:
        Data File.
    """
    cancer_data = load_breast_cancer()
    feature_name = cancer_data['feature_names']
    head = '\t'.join(feature_name)+'target'
    with open(datafil,'w') as out:
        out.writelines(head+'\n')
        for index,i in enumerate(cancer_data['data']):
            tmp_data = map(str,i)                              # a sample data
            tmp_target = str(cancer_data['target'][index])     # a sample target
            line = '\t'.join(tmp_data)+'\t'+tmp_target
            out.writelines(line+'\n')
    return


def Data2Matrix(datafil):
    """
        将文本数据转化为矩阵。
    :param datafil: General data file. text format.
    :return:
        data_matrix: data_matrix
        target_matrix: target_matrix
    """
    data_matrix = []
    target_matrix = []
    with open(datafil,'r') as f:
        for line in islice(f,1,None):
            line = line.split()
            line = map(float,line)
            data_matrix.append(line[:-1])
            target_matrix.append(line[-1])

    data_matrix = np.array(data_matrix)
    target_matrix = np.array(target_matrix)

    return data_matrix, target_matrix


def KNN_classify(inX,dataSet,labels,K):
    """
        K近邻算法分类函数，使用欧拉公式。
    :param inX: 未知类别数据
    :param dataSet: 已知类别数据集特征数据
    :param labels: 已知类别数据集类别数据
    :param K: K points
    :return:
    """
    # distance handle
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDisIndices = distances.argsort()

    # select the most k nearby distance.
    classCount = {}
    for i in range(K):
        votelabel = labels[sortedDisIndices[i]]
        classCount[votelabel] = classCount.get(votelabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)

    return sortedClassCount[0][0]


def NormalizeData(dataSet):
    """
        Normalize Data into 0-1
    :param dataSet: DataSet
    :return:
    """
    minValue = dataSet.min(0)
    maxValue = dataSet.max(0)
    ranges = maxValue - minValue
    m = dataSet.shape[0]
    normData = dataSet - np.tile(minValue,(m,1))
    normData = normData / np.tile(ranges,(m,1))
    return normData, ranges, minValue


def KNNTest2(datafil,K=1):
    """
        KNN Test2
    :param datafil: data file
    :param K: K point
    :return: KNN score
    """
    data_M, data_T = Data2Matrix(datafil)
    train_X,test_X,train_Y,test_Y = train_test_split(data_M,data_T,random_state=0)
    train_X = NormalizeData(train_X)[0]
    test_X = NormalizeData(test_X)[0]
    correct = 0
    error = 0
    for index in range(len(test_X)):
        predict_label = KNN_classify(test_X[index],train_X,train_Y,K)
        if predict_label == test_Y[index]:
            correct += 1
        else:
            error += 1
    return correct / float(correct+error)


if __name__ == '__main__':
    # KNN by sklearn
    KNNTest()

    # KNN by self-defined
    SKLearnData2File('breast_cancer.txt')
    KNNTest2('breast_cancer.txt')