
import json
import math
import time
import numpy as np
import pandas as pd
import geojson
import csv
from shapely.geometry import mapping, shape, Polygon, MultiPoint
import shapely.wkt as wkt
import folium
def getPOIFromCSVtoGraph(filePath):
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
                    # 判断经纬度是否为 00 0 0 如 果为0就说明这条记录无效
                    if (line[12]) == "":
                        continue
                    if abs(float(line[10])) < +1e-6 and abs(float(line[11])) <= 1e-6 and abs(
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
                        if (EmptyPOI == 159219):
                            continue

                        # G.nodes[EmptyPOI]["dropOffFrequency"] = (
                        #             nx.get_node_attributes(G, "dropOffFrequency")[EmptyPOI] + 1)

                        G.nodes[EmptyPOI]["totalVisitFrequency"] = (
                                nx.get_node_attributes(G, "totalVisitFrequency")[EmptyPOI] + 1)
                        PickupPOI = index[1]
                        if (PickupPOI == 159219):
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
    for line in f.readlines():
        itemsCount = itemsCount + 1
        # if count!=1:
        lines = line.strip().split(",")
        if lines[8] in locationIDToPOIMap.keys():
            if locationIDToPOIMap.get(lines[8])[4]<lines[4]:
                locationIDToPOIMap[lines[8]]=lines
        else:
            locationIDToPOIMap[lines[8]]=lines
    f.close()
    with open("E:/data/ExprimentField/manhatan/Poi_NYC_Manhatan_GridDivide2.csv", "w",newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["IDcode", "name", "Type","UseType","Check-in","UseNumber","latitude","longitude"])
        for key in locationIDToPOIMap.keys():
            writer.writerow(locationIDToPOIMap.get(key))

    end_time = time.time()
    cost_time = (end_time - start_time)

    print("The number of items:" + str(itemsCount))
    print("The number of invalid items:" + str(invalidItemNumber))
    print('Total time spent on loading POI data {:.5f} second.'.format(cost_time))
    print("--------Done!--------")



filePath = "E:/data/ExprimentField/manhatan/Poi_NYC_Manhatan_GridDivide1.csv"
getPOIFromCSVtoGraph(filePath)

