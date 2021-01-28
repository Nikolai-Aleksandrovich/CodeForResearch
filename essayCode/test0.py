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
def getEdgesFromCSVtoGraph(filePath, G, lontitudelatitudeArray, mytree, IfPrint):
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
                    # 输出每一辆车按时间排列的行驶轨迹
                    if IfPrint == 1:
                        print("--------" + str(currentCarID) + "----------------------------------")
                        # print(SortedcurrentCarArray)

                    # 得到按照时间排列的每一个车辆的行驶轨迹后算边权重
                    for j in range(0, len(SortedcurrentCarArray) - 1):
                        # print ""
                        first = SortedcurrentCarArray[j]
                        second = SortedcurrentCarArray[j + 1]

                        # print ""

                        # 空车时间，计算指向哪个时间节点
                        # EmptyTime = first[6]
                        # pickUpTime = second[5]
                        #
                        # get_time = str(EmptyTime).rpartition(' ')[-1]
                        # # print(get_time)
                        # get_hour1 = str(get_time).rpartition(':')[0]
                        # get_hour = str(get_hour1).rpartition(':')[0]
                        #
                        # a = int(get_hour)
                        # if a > 5 and a < 8:  # G0:6:00-7:59  G1:8:00-11:59  G2:12:00-13:59  G3:14-17  G4:17-19
                        # G5:19-22  G6:22-next6 G = listG[0] elif a > 7 and a < 12: G = listG[1] elif a > 11 and a < 14:
                        # G = listG[2] elif a > 13 and a < 18: G = listG[3] elif a > 17 and a < 20: G = listG[4] elif a >
                        # 19 and a < 22: G = listG[5] else: G = listG[6]

                        # 判断是哪个时间节点-----------------------------------

                        # a2=a+1
                        # #向上取整
                        # idvalue=int(math.ceil(float(a2)/2))
                        # TimeNodeID1="Time"+str(idvalue)
                        # current_timeslot_node=G.nodes(TimeNodeID1)


                        if d == 0:
                            d = 1

                        n = 1  # 该时间片内从下一条起点出发的旅途数量

                        # 空车和接到客人的位置信息
                        EmptyAndpickUpPosition = [[float(first[12]), float(first[13])],
                                                  [float(second[10]), float(second[11])]]
                        EmptyAndpickUpPositionArray = np.array(EmptyAndpickUpPosition)

                        # 在KDTree中查找空车和接到客人的位置最近的POI，返回最近距离和POI的ID

                        # print distance, index

                        # TimeNode=0
                        # 从前一条轨迹中得到信息增益
                        # print int(index[1])
                        # print G.nodes[int(index[1])]
                        # InformationEntropy = G.nodes[int(index[1])]["InformationEntropy"]
                        # if IfPrint == 1:
                        #     print("InformationEntropy:" + str(InformationEntropy))

                        # 可调参数
                        alpha = 100
                        beta = 1
                        current_total_x = 0
                        current_total_d = 0
                        current_total_trip = 0


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

        print("The number of all nodes:" + str(G.number_of_nodes()))
        print("The number of edges from nx:" + str(G.number_of_edges()))
        print("The number of edges from my calculate:" + str(EdgesCount))
        print("The number of valid edges:" + str(ValidCount))
        print('Total time spent on loading car data {:.5f} second.'.format(cost_time))
        print("--------Done!--------")
    # nx.write_weighted_edgelist(G, 'Edge.weighted.edgelist', comments='#', delimiter=' ', encoding='utf-8')
    return G

