# coding:utf-8
import csv
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


def getPOIFromCSVtoGraph(filePath, G):
    start_time = time.time()
    print("--------Add POI Nodes--------")
    try:
        f = open(filePath)
    except IOError:
        print("---**---打开文件失败！请检查是否存在该文件！并重新输入文件名!---**---")
    print("filePath:", filePath)

    itemsCount = 0
    NodesCount = 0
    repeatedItemsCount = 0
    invalidItemNumber = 0
    lontitudelatitudeList = []
    for line in f.readlines():
        itemsCount = itemsCount + 1
        # if count!=1:
        lines = line.strip().split(",")
        if len(lines) == 9:
            NodesCount = NodesCount + 1
            # 将图中的POI节点的经纬度坐标放进列表中
            lontitudelatitudeList.append([float(lines[7]), float(lines[6])])
        else:
            # print lines
            invalidItemNumber = invalidItemNumber + 1
    f.close()

    end_time = time.time()
    cost_time = (end_time - start_time)
    print('Total time spent on loading POI data {:.5f} second.'.format(cost_time))
    print("--------Done!--------")
    lontitudelatitudeArray = np.array(lontitudelatitudeList)
    # nx.write_weighted_edgelist(G, 'POI.weighted.edgelist', comments='#', delimiter=' ', encoding='utf-8')
    return G, lontitudelatitudeArray


def getEdgesFromCSVtoGraph(filePath, lontitudelatitudeArray, mytree):
    global FinalList
    EmptyList = []
    PickUpList = []

    start_time = time.time()
    print("--------Add Edges--------")
    chunksize = 10 ** 6
    for chunk in pd.read_csv(filePath, chunksize=chunksize):

        chunkList = chunk.values.tolist()
        EdgesCount = 0
        itemsCount = 0
        edgesList = []
        for line in chunkList:
            itemsCount = itemsCount + 1
            if itemsCount != 1:
                # lines = line.strip().split(",")
                EdgesCount = EdgesCount + 1
                edgesList.append(line)
        ValidCount = 0
        FinalList = edgesList
        for i in range(0, len(edgesList) - 1):
            test = edgesList[i]
            EmptyAndpickUpPosition = [[float(test[7]), float(test[8])],
                                      [float(test[5]), float(test[6])]]
            EmptyAndpickUpPositionArray = np.array(EmptyAndpickUpPosition)

            # 在KDTree中查找空车和接到客人的位置最近的POI，返回最近距离和POI的ID
            distance, index = queryPoint(mytree, EmptyAndpickUpPositionArray)
            FinalList[i].append(index[0])
            FinalList[i].append(index[1])
        end_time = time.time()
        cost_time = (end_time - start_time)
        with open("E:/data/ExprimentField/test/timeSlot0.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(
            #     ["medallion", "hack_license", "vendor_id", "rate_code", "store_and_fwd_flag", "pickup_datetime",
            #      " dropoff_datetime", " passenger_count", " trip_time_in_secs", " trip_distance", " pickup_longitude",
            #      " pickup_latitude", " dropoff_longitude", " dropoff_latitude"])
            for row in FinalList:
                writer.writerow(row)


        print('Total time spent on loading car data {:.5f} second.'.format(cost_time))
        print("--------Done!--------")
    # nx.write_weighted_edgelist(G, 'Edge.weighted.edgelist', comments='#', delimiter=' ', encoding='utf-8')
    return FinalList

    # --------------------以下函数是用于寻找经纬度坐标最近的POI点----------


def changedata(data):
    R = 6367
    phi = np.deg2rad(data[:, 1])  # LAT
    theta = np.deg2rad(data[:, 0])  # LON
    # 转化过的数据一共有五列 Lontitude Latitude 转换后的笛卡尔坐标
    data = np.c_[data, R * np.cos(phi) * np.cos(theta), R * np.cos(phi) * np.sin(theta), R * np.sin(phi)]
    return data


def deleteNode(G, number):
    print("--------Delete POI Nodes--------")
    removeList = []
    for node, attr in G.nodes(data=True):
        frequency = attr['totalVisitFrequency']
        if frequency < number:
            removeList.append(node)
    print("The number of deleted nodes:" + str(len(removeList)))
    G.remove_nodes_from(removeList)
    print("The number of all nodes:" + str(G.number_of_nodes()))
    print("The number of edges:" + str(G.number_of_edges()))
    print("--------End of Delete POI Nodes--------")
    # nx.write_weighted_edgelist(G, 'delete.weighted.edgelist', comments='#', delimiter=' ', encoding='utf-8')
    return G


def creatKdTree(ref_data):
    ref_data = changedata(ref_data)
    # print ref_data
    tree = scipy.spatial.cKDTree(ref_data[:, 2:5])
    return tree
    # Convert Euclidean chord length to great circle arc length


def dist_to_arclength(chord_length):
    R = 6367
    central_angle = 2 * np.arcsin(chord_length / (2.0 * R))
    arclength = R * central_angle
    return arclength


def queryPoint(tree, que_Data):
    que_Data = changedata(que_Data)
    distance, index = tree.query(que_Data[:, 2:5])
    return dist_to_arclength(distance), index


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


G = nx.DiGraph()

# 创建图中的一类节点 为POI bipartite=0
filePath = "E:/data/ExprimentField/manhatan/Poi_NYC_Manhatan_DBSCAN.csv"
G, lontitudelatitudeArray = getPOIFromCSVtoGraph(filePath, G)
# print lontitudelatitudeArray
# 创建POI坐标的KDTree
mytree = creatKdTree(lontitudelatitudeArray)

filePathtest = "E:/data/ExprimentField/test/newtimeSlot0.csv"
filePath1 = "./data/trip_data_1.csv"
filePath2 = "./data/trip_data_2.csv"
filePath3 = "./data/trip_data_3.csv"
filePath4 = "./data/trip_data_4.csv"
filePath5 = "./data/trip_data_5.csv"
filePath6 = "./data/trip_data_6.csv"
IfPrint = 0

FinalList = getEdgesFromCSVtoGraph(filePathtest, lontitudelatitudeArray, mytree)


# G = getEdgesFromCSVtoGraph(filePath1, G, lontitudelatitudeArray, mytree, IfPrint)
# G = getEdgesFromCSVtoGraph(filePath2, G, lontitudelatitudeArray, mytree, IfPrint)
# G = getEdgesFromCSVtoGraph(filePath3, G, lontitudelatitudeArray, mytree, IfPrint)
# G = getEdgesFromCSVtoGraph(filePath4, G, lontitudelatitudeArray, mytree, IfPrint)
# G = getEdgesFromCSVtoGraph(filePath5, G, lontitudelatitudeArray, mytree, IfPrint)
# G = getEdgesFromCSVtoGraph(filePath6, G, lontitudelatitudeArray, mytree, IfPrint)
# print(nx.number_of_edges(G))


# np.savez("adjex.npz",A)
# nx.write_edgelist(G, "./data/test.edgelist", comments='#', delimiter=' ', data=['weight'], encoding='utf-8')
