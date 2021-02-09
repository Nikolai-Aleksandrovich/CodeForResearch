
import time
import networkx as nx

import sklearn.metrics.pairwise as pw
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
# coding:utf-8
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import scipy.spatial
import math
import time
import datetime
from math import radians, cos, sin, asin, sqrt
import dask.dataframe as dd
import pandas as pd
from collections import namedtuple


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """

    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])


def precision_and_recall(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits / (1.0 * len(ranked_list) if len(ground_list) != 0 else 1)
    rec = hits / (1.0 * len(ground_list) if len(ground_list) != 0 else 1)
    return pre, rec


def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)


def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

    idcg = getDCG(np    .sort(relevance)[::-1])

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg

def fare(distance):
    return (5.3+0.5*distance)

def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0

def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    # method 2
    # cos = bit_product_sum(x, y) / (np.sqrt(bit_product_sum(x, x)) * np.sqrt(bit_product_sum(y, y)))

    # method 3
    # dot_product, square_sum_x, square_sum_y = 0, 0, 0
    # for i in range(len(x)):
    #     dot_product += x[i] * y[i]
    #     square_sum_x += x[i] * x[i]
    #     square_sum_y += y[i] * y[i]
    # cos = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


def getCosineSimilarityFromOut(Bottom15Data, Top15trainResult,IdToGPSMapFile,Bottom15Income, k):
    global IDCount
    start_time = time.time()
    # top15 Result.txt
    try:
        f = open(Top15trainResult)
    except IOError:
        print("---**---打开文件失败！请检查是否存在该文件！并重新输入文件名!---**---")
    print("filePath:", Top15trainResult)
    TrainResultCount = 0
    EdgeListNodeList = []
    for line in f.readlines():
        TrainResultCount = TrainResultCount + 1

        lines = (line.strip().split(" "))
        EdgeListNodeList.append(lines)

    TrainResult = np.array(EdgeListNodeList, dtype=float)
    f.close()
    #print('Top15Result',TrainResult)

    # Bottom15Data
    try:
        f = open(Bottom15Data)
    except IOError:
        print("---**---打开文件失败！请检查是否存在该文件！并重新输入文件名!---**---")
    print("filePath:", Bottom15Data)
    TestitemsCount = 0
    Data = []
    for line in f.readlines():
        TestitemsCount = TestitemsCount + 1
        lines = line.strip().split(" ")
        Data.append(lines)
    Bottom15Data = np.array(Data)
    f.close()
    #print('Bottom15data',Bottom15Data)
    # 得到最弱的15

    # 累加收入
    try:
        f = open(Bottom15Income)
    except IOError:
        print("---**---打开文件失败！请检查是否存在该文件！并重新输入文件名!---**---")
    print("filePath:", Bottom15Income)
    TestitemsCount = 0
    Income = []
    for line in f.readlines():
        TestitemsCount = TestitemsCount + 1
        lines = line.strip().split('/n')
        Income.append(lines)
    Bottom15Income = np.array(Income, dtype=float)
    TotalBottom15Income=0
    print(len(Bottom15Income))
    b=0
    for list in Bottom15Income:
        b=b+1
        TotalBottom15Income=float(list[0])+TotalBottom15Income

    f.close()
    # print('TotalBottom15Income',TotalBottom15Income)
    # 得到最弱的15

    # 测试数据的edgelist文件
    try:
        f = open(IdToGPSMapFile)
    except IOError:
        print("---**---打开文件失败！请检查是否存在该文件！并重新输入文件名!---**---")
    print("filePath:", IdToGPSMapFile)
    IdToGPSDict= {}
    for line in f.readlines():
        lines = line.strip().split(",")
        if lines[8] not in IdToGPSDict:
            IdToGPSDict[lines[8]]=[float(lines[6]),float(lines[7])]
    f.close()
    # print('idToGPSDict',IdToGPSDict)
    #得到字典{ID:[la,long]},'3116': [40.720914, -74.001493]
    trainNodeID = []  # 较大训练数据集节点ID

    finalCSList = [[] for i in range(TrainResultCount)]
    a = 0
    # for line in Bottom15Data:
    #
    #     print('Bottom',line[0])
    # for line in TrainResult:
    #     print('TrainResult',line[0])
    for trainArray in TrainResult:

        CSListForEachRow = []
        trainNodeID.append(trainArray[0])
        trainVector = np.delete(trainArray, 0)
        for otherArray in TrainResult:
            if ((otherArray==trainArray).all()):
                continue
            otherVector=np.delete(otherArray, 0)
            CSListForEachRow.append(cos_sim(trainVector, otherVector))  # 每行的余弦相似性加入列表
        finalCSList[a] = CSListForEachRow
        a=a+1
    # print('start print finalCSList')
    # print(finalCSList)
    # print('end of finalCSList')
    TopKIndex = [[] for i in range(TrainResultCount)]
    for i in range(0, TrainResultCount):
        temp = finalCSList[i]
        for j in range(0, k):
            TopKIndex[i].append(temp.index(max(temp)))
            temp[temp.index(max(temp))] = 0.001
    # print('TopKIndex',TopKIndex)
    TotalIncome = 0
    TotalDistance = 0
    TotalTime = 0
    RecommendCount=0
    realCount = TrainResultCount
    for line1 in Bottom15Data:
        # print('line1',line1[0])
        RecommendDestID = []
        currentStartID = line1[0]
        ToThisPlaceCount = line1[2]
        index = 0
        for line2 in TrainResult:
            # print('line2',line2[0])
            if (line2[0] == line1[0]) and ToThisPlaceCount!=0:
                tempID=TopKIndex[index][ToThisPlaceCount-1]
                RecommendDestID.append(tempID)
                index=index+1
                ToThisPlaceCount=ToThisPlaceCount-1
        # print('RecommendID',RecommendDestID)
        # print('CurrentID',currentStartID)
        for i in range(len(RecommendDestID)):
            StartLa = IdToGPSDict[currentStartID][0]
            StartLong = IdToGPSDict[currentStartID][1]
            EndLa = IdToGPSDict[RecommendDestID[i]][0]
            EndLong = IdToGPSDict[RecommendDestID[i]][1]
            distance = haversine(StartLong, StartLa, EndLong, EndLa)
            TotalDistance=distance+TotalDistance
            TotalIncome=fare(distance*0.62137)+TotalIncome
            print('startLA',StartLa)
            print('SratLONG',StartLong)
            print('EndLa',EndLa)
            print('EndLong',EndLong)
            print('Distance',distance)
            print('TotalDistance',TotalDistance)
            print('TotalIncome',TotalIncome)


    print(TrainResultCount)
    print(realCount)
    end_time = time.time()
    cost_time = (end_time - start_time)
    print('TotalIncome',TotalIncome)
    print('Total time spent on loading car data {:.5f} second.'.format(cost_time))
    TotalTime = TotalDistance*0.62137/7.5
    return TotalIncome,TotalDistance,TotalTime,TotalBottom15Income


Bottom15Data = "D:/data/ExperimentField/TaxiDataWithFare/feb/feb1/test2.weighted.edgelist"
Top15trainResult = "D:/data/ExperimentField/TaxiDataWithFare/feb/feb1/result.txt"
Bottom15Income= "D:/data/ExperimentField/TaxiDataWithFare/feb/feb1/Bottom15Income.csv"
IdToGPSMapFile = "D:/data/ExperimentField/test/Poi_NYC_Manhatan_GridDivideFinal.csv"



TotalIncome,TotalDistance,TotalTime,TotalBottom15Income = getCosineSimilarityFromOut(Bottom15Data, Top15trainResult,IdToGPSMapFile,Bottom15Income, 3)
print(TotalIncome,TotalDistance,TotalTime,TotalBottom15Income  )
