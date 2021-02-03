
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
        # if NodesCount in G:
        #     repeatedItemsCount = repeatedItemsCount + 1
        #     continue
            # print True
        if len(lines) == 9:

            # 求该POI的信息熵
            if float(lines[4]) < 1:
                continue

            # c = float(lines[4])
            # if float(lines[5]) < 1e-6:
            #     n = 1
            # else:
            #     n = float(lines[5])

            # InformationEntropyValue = (c / n) * math.log(n)

            # G.add_node(NodesCount,name=lines[0],checkInCount=lines[4],
            #     checkInUsersNumebr=lines[5],latitude=float(lines[6]),
            #     lontitude=float(lines[7]),bipartite=0,InformatioEntropy=)

            # G.add_node(NodesCount, name=lines[0], latitude=float(lines[6]),
            #            lontitude=float(lines[7]),
            #            InformationEntropy=InformationEntropyValue, pickUpFrequency=0, dropOffFrequency=0,
            #            totalVisitFrequency=0)
            G.add_node(NodesCount,totalVisitFrequency=0)
            # G.add_node(NodesCount, totalVisitFrequency=0, informationEntropy=InformationEntropyValue)

            NodesCount = NodesCount + 1

            # 将图中的POI节点的经纬度坐标放进列表中
            lontitudelatitudeList.append([float(lines[7]), float(lines[6])])
        else:
            print (lines)
            invalidItemNumber = invalidItemNumber + 1
    f.close()

    end_time = time.time()
    cost_time = (end_time - start_time)

    print("The number of items:" + str(itemsCount))
    print("The number of invalid items:" + str(invalidItemNumber))
    print("The number of all nodes:" + str(G.number_of_nodes()))
    print("The number of edges:" + str(G.number_of_edges()))
    print("The number of repeated items:" + str(repeatedItemsCount))
    print('Total time spent on loading POI data {:.5f} second.'.format(cost_time))
    print("--------Done!--------")
    lontitudelatitudeArray = np.array(lontitudelatitudeList)
    # nx.write_weighted_edgelist(G, 'POI.weighted.edgelist', comments='#', delimiter=' ', encoding='utf-8')
    return G, lontitudelatitudeArray


def getAttrFromCSVtoGraph(filePath, G, lontitudelatitudeArray, mytree, IfPrint):
    # 逐条计算旅途，为一个节点添加“度”属性的值
    start_time = time.time()
    print("--------Add Attr--------")
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


                    # 得到按照时间排列的每一个车辆的行驶轨迹后算边权重
                    for j in range(0, len(SortedcurrentCarArray) - 1):
                        # print ""
                        first = SortedcurrentCarArray[j]
                        second = SortedcurrentCarArray[j + 1]


                        # 空车和接到客人的位置信息
                        EmptyAndpickUpPosition = [[float(first[12]), float(first[13])],
                                                  [float(second[10]), float(second[11])]]
                        EmptyAndpickUpPositionArray = np.array(EmptyAndpickUpPosition)

                        # 在KDTree中查找空车和接到客人的位置最近的POI，返回最近距离和POI的ID
                        distance, index = queryPoint(mytree, EmptyAndpickUpPositionArray)
                        EmptyPOI = index[0]
                        if(EmptyPOI==159219):
                            continue

                        # G.nodes[EmptyPOI]["dropOffFrequency"] = (
                        #             nx.get_node_attributes(G, "dropOffFrequency")[EmptyPOI] + 1)

                        G.nodes[EmptyPOI]["totalVisitFrequency"] = (
                                    nx.get_node_attributes(G, "totalVisitFrequency")[EmptyPOI] + 1)
                        PickupPOI = index[1]
                        if(PickupPOI == 159219):
                            continue

                        # G.nodes[PickupPOI]["pickUpFrequency"] = (
                        #             nx.get_node_attributes(G, "pickUpFrequency")[PickupPOI] + 1)
                        G.nodes[PickupPOI]["totalVisitFrequency"] = (
                                    nx.get_node_attributes(G, "totalVisitFrequency")[PickupPOI] + 1)
                        # print distance, index

                currentCarID = i[0]  # SortedCarList.append(SortedcurrentCarArray))#一个新的CarID)
                currentCarList = [i]

        end_time = time.time()
        cost_time = (end_time - start_time)

        print("The number of all nodes:" + str(G.number_of_nodes()))
        print("The number of edges:" + str(G.number_of_edges()))
        print('Total time spent on loading car data {:.5f} second.'.format(cost_time))
        print("--------Done!--------")
    # nx.write_weighted_edgelist(G, 'Attr.weighted.edgelist', comments='#', delimiter=' ', encoding='utf-8')
    return G


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

                        # 其余Weight计算参数
                        x = float(second[9])  # 下一条旅行记录的收入
                        d = haversine(float(first[12]), float(first[13]), float(second[10]),
                                      float(second[11]))  # 一条旅行记录的路程+空车寻客路程
                        if d == 0:
                            d = 1

                        n = 1  # 该时间片内从下一条起点出发的旅途数量

                        # 空车和接到客人的位置信息
                        EmptyAndpickUpPosition = [[float(first[12]), float(first[13])],
                                                  [float(second[10]), float(second[11])]]
                        EmptyAndpickUpPositionArray = np.array(EmptyAndpickUpPosition)

                        # 在KDTree中查找空车和接到客人的位置最近的POI，返回最近距离和POI的ID
                        distance, index = queryPoint(mytree, EmptyAndpickUpPositionArray)
                        EmptyPOI = index[0]

                        PickupPOI = index[1]

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

                        if G.has_edge(EmptyPOI, PickupPOI):
                            # 存在边就对边的值进行权重加和
                            # current_total_x = nx.get_edge_attributes(G, 'current_total_x') + x
                            # current_total_trip = nx.get_edge_attributes(G,'current_total_trip') + n
                            #  计算复杂权重的权重语句
                            # G.add_edge(index[0], index[1], weight=G.edges[index[0], index[1]]['weight'] +
                            #            n * InformationEntropy * (
                            #             alpha * x /
                            #             n) / (
                            #                     d * beta))

                            G.add_edge(EmptyPOI, PickupPOI,
                                       weight=G.get_edge_data(EmptyPOI, PickupPOI).get('weight') + 1)  # 添加的权重语句，只计算频率
                            # G.edges[index[0], index[1]]['current_total_x'] = G.edges[index[0], index[1]][
                            #                                                         'current_total_trip'] + x
                            # G.edges[index[0], index[1]]['current_total_trip'] = G.edges[index[0], index[1]]['current_total_trip'] + n
                            # G.edges[index[0], index[1]] ['weightOfEdge'] = (
                            #         G.edges[index[0], index[1]]['current_total_trip'] * InformationEntropy * (
                            #         alpha * G.edges[index[0], index[1]]['current_total_x'] /
                            #         G.edges[index[0], index[1]]['current_total_trip']) / (
                            #                 G.edges[index[0], index[1]]['current_total_d'] * beta))

                        else:
                            # 不存在边就添加边
                            #  G.add_edge(index[0], index[1],
                            #             weight=(n * InformationEntropy * alpha * x / n) / (d * beta))
                            G.add_edge(EmptyPOI, PickupPOI, weight=(1))  # 添加的权重语句，只计算频率
                        # if index[0]==index[1]:
                        #     G.remove_edge(index[0], index[1])
                        # if G.edges[index[0], index[1]]['weight'] == 0:
                        #   G.edges[index[0], index[1]]['weight'] = 0.0000001
                        # G.add_edge(index[0], index[1], current_total_x=x, current_total_d=d, current_total_trip=n,
                        #            weightOfEdge=(n * InformationEntropy * alpha * x / n) / (d * beta))
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
    for node,attr in G.nodes(data=True):
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
POIFilePath = "E:/data/ExprimentField/test/POIData/Poi_NYC_Manhatan_GridDivideFinal.csv"
G, lontitudelatitudeArray = getPOIFromCSVtoGraph(POIFilePath, G)
# print lontitudelatitudeArray
# 创建POI坐标的KDTree
mytree = creatKdTree(lontitudelatitudeArray)

# 添加图中的另一类节点 为时间节点 bipartite=1
# G=gd.getTimeNodestoGraph(filePath,G)

# nx.draw_networkx(G)
Jan0TaxiFilePathTrain = "E:/data/ExprimentField/test/jan/jan0/train.csv"
Jan1TaxiFilePathTrain = "E:/data/ExprimentField/test/jan/jan1/train.csv"
Jan2TaxiFilePathTrain = "E:/data/ExprimentField/test/jan/jan2/train.csv"
Jan3TaxiFilePathTrain = "E:/data/ExprimentField/test/jan/jan3/train.csv"
Jan4TaxiFilePathTrain = "E:/data/ExprimentField/test/jan/jan4/train.csv"
Jan5TaxiFilePathTrain = "E:/data/ExprimentField/test/jan/jan5/train.csv"
Jan6TaxiFilePathTrain = "E:/data/ExprimentField/test/jan/jan6/train.csv"
Feb0TaxiFilePathTrain = "E:/data/ExprimentField/test/feb/feb0/train.csv"
Feb1TaxiFilePathTrain = "E:/data/ExprimentField/test/feb/feb1/train.csv"

Jan0TaxiFilePathTest = "E:/data/ExprimentField/test/jan/jan0/test.csv"
Jan1TaxiFilePathTest = "E:/data/ExprimentField/test/jan/jan1/test.csv"
Jan2TaxiFilePathTest = "E:/data/ExprimentField/test/jan/jan2/test.csv"
Jan3TaxiFilePathTest = "E:/data/ExprimentField/test/jan/jan3/test.csv"
Jan4TaxiFilePathTest = "E:/data/ExprimentField/test/jan/jan4/test.csv"
Jan5TaxiFilePathTest = "E:/data/ExprimentField/test/jan/jan5/test.csv"
Jan6TaxiFilePathTest = "E:/data/ExprimentField/test/jan/jan6/test.csv"
Feb0TaxiFilePathTest = "E:/data/ExprimentField/test/feb/feb0/test.csv"
Feb1TaxiFilePathTest = "E:/data/ExprimentField/test/feb/feb1/test.csv"

IfPrint = 0
# G = getAttrFromCSVtoGraph(filePathtest, G, lontitudelatitudeArray, mytree, IfPrint)
# G = getAttrFromCSVtoGraph(filePath1, G, lontitudelatitudeArray, mytree, IfPrint)
# G = getAttrFromCSVtoGraph(filePath2, G, lontitudelatitudeArray, mytree, IfPrint)
# G = getAttrFromCSVtoGraph(filePath3, G, lontitudelatitudeArray, mytree, IfPrint)
# G = getAttrFromCSVtoGraph(filePath4, G, lontitudelatitudeArray, mytree, IfPrint)
# G = getAttrFromCSVtoGraph(filePath5, G, lontitudelatitudeArray, mytree, IfPrint)
# # G = getAttrFromCSVtoGraph(filePath6, G, lontitudelatitudeArray, mytree, IfPrint)
# G = deleteNode(G, 0)
G = getEdgesFromCSVtoGraph(Feb1TaxiFilePathTest , G, lontitudelatitudeArray, mytree, IfPrint)
# G = getEdgesFromCSVtoGraph(filePath1, G, lontitudelatitudeArray, mytree, IfPrint)
# G = getEdgesFromCSVtoGraph(filePath2, G, lontitudelatitudeArray, mytree, IfPrint)
# G = getEdgesFromCSVtoGraph(filePath3, G, lontitudelatitudeArray, mytree, IfPrint)
# G = getEdgesFromCSVtoGraph(filePath4, G, lontitudelatitudeArray, mytree, IfPrint)
# G = getEdgesFromCSVtoGraph(filePath5, G, lontitudelatitudeArray, mytree, IfPrint)
# G = getEdgesFromCSVtoGraph(filePath6, G, lontitudelatitudeArray, mytree, IfPrint)
# print(nx.number_of_edges(G))


# np.savez("adjex.npz",A)
# nx.write_edgelist(G, "./data/test.edgelist", comments='#', delimiter=' ', data=['weight'], encoding='utf-8')
nx.write_weighted_edgelist(G, 'test2.weighted.edgelist', comments='#', delimiter=' ', encoding='utf-8')