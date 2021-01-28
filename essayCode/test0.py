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
def getEdgesFromCSVtoGraph(filePath,IfPrint):
    # 添加权重
    start_time = time.time()
    print("--------Add Edges--------")
    chunksize = 10 ** 6
    for chunk in pd.read_csv(filePath, chunksize=chunksize):
        chunkList = [[]]
        chunkList = chunk.values.tolist()
        EdgesCount = 0
        itemsCount = 0
        edgesList = []
        for line in chunkList:
            itemsCount = itemsCount + 1
            if itemsCount != 1:
                # lines = line.strip().split(",")
                if len(line) == 14:
                    # 判断经纬度是否为 00 0 0 如 果为0就说明这条记录无效
                    if (line[12]) == "":
                        continue
                    if abs(float(line[10])) < 1e-6 and abs(float(line[11])) <= 1e-6 and abs(
                            float(line[12])) <= 1e-6 and abs(float(line[13])) <= 1e-6:
                        1
                    else:
                        EdgesCount = EdgesCount + 1
                        edgesList.append(line)
        edgesArray = np.array(edgesList)
        # 对其按照第1列排序
        SortedEdgesArray = edgesArray[edgesArray[:, 0].argsort()]
        # 打印排好序的数组
        # print(SortedEdgesArray)
        # #打印原数组
        # print(edgesArray)

        currentCarID = ""
        currentCarList = []
        SortedCarList = []
        ValidCount = 0
        for i in SortedEdgesArray:
            # print i[0]
            # print "currentCarID:"+str(currentCarID)
            if currentCarID == i[0]:
                # 一个车的连续轨迹
                # print currentCarID,i[5],i[6]
                currentCarList.append(i)
            else:
                # print currentCarList

                if len(currentCarList) != 0 and len(currentCarList) != 1:
                    # 如果车辆的行驶轨迹只有一条数据，那么这一条算无效数据
                    ValidCount = ValidCount + len(currentCarList)
                    currentCarArray = np.array(currentCarList)
                    # print currentCarArray
                    # 对其按照第6列排序
                    SortedcurrentCarArray = currentCarArray[currentCarArray[:, 5].argsort()]
                    if len(SortedcurrentCarArray)>2:
                        print(currentCarID)
                        print(SortedcurrentCarArray)
                    # 输出每一辆车按时间排列的行驶轨迹
                    # if IfPrint == 1:
                    #     print("--------" + str(currentCarID) + "----------------------------------")
                    #     print(SortedcurrentCarArray)

                    # 得到按照时间排列的每一个车辆的行驶轨迹后算边权重
                currentCarID = i[0]  # SortedCarList.append(SortedcurrentCarArray))#一个新的CarID)
                currentCarList = [i]

        # 对最后一个array进行排序
        if len(currentCarList) != 0 and len(currentCarList) != 1:
            # 如果车辆的行驶轨迹只有一条数据，那么这一条算无效数据
            ValidCount = ValidCount + len(currentCarList)
            currentCarArray = np.array(currentCarList)
            # print currentCarArray
            # 对其按照第6列排序
            SortedcurrentCarArray = currentCarArray[currentCarArray[:, 5].argsort()]
            if IfPrint == 1:
                print("--------" + str(currentCarID) + "-----------------")
                print(SortedcurrentCarArray)

            # SortedCarList.append(SortedcurrentCarArray)
        # SortedCarArray=np.array(SortedCarList)
        # print SortedCarArray

        end_time = time.time()
        cost_time = (end_time - start_time)

        print("The number of edges from my calculate:" + str(EdgesCount))
        print("The number of valid edges:" + str(ValidCount))
        print('Total time spent on loading car data {:.5f} second.'.format(cost_time))
        print("--------Done!--------")
    # nx.write_weighted_edgelist(G, 'Edge.weighted.edgelist', comments='#', delimiter=' ', encoding='utf-8')
    return G
filePathtest = "E:/data/ExprimentField/test/10%of10%.csv"
G = getEdgesFromCSVtoGraph(filePathtest,1)
