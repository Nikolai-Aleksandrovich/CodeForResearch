
import json
import math
import time
import numpy as np
import geojson
import csv
from shapely.geometry import mapping, shape, Polygon, MultiPoint
import shapely.wkt as wkt
import folium
def getPOIFromCSVtoGraph(filePath):
    start_time = time.time()
    print("--------Loading POI data--------")
    try:
        f = open(filePath)
    except IOError:
        print("---**---打开文件失败！请检查是否存在该文件！并重新输入文件名!---**---")
    print("filePath:", filePath)

        # 写入多行用writerows
    itemsCount = 0
    NodesCount = 0
    repeatedItemsCount = 0
    invalidItemNumber = 0
    validateList = [[]]
    for line in f.readlines():
        itemsCount = itemsCount + 1
        # if count!=1:
        lines = line.strip().split(";")
        if len(lines) == 8:
            # 求该POI的信息熵
            if(judgerOfGps(float(lines[7]), float(lines[6]))):
                validateList.append(lines)
            else:
                invalidItemNumber = invalidItemNumber + 1

        else:
            invalidItemNumber = invalidItemNumber + 1
    f.close()
    with open("E:/data/ExprimentField/Poi_NYC_Manhatan.csv", "w",newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["IDcode", "name", "Type","UseType","Check-in","UseNumber","latitude","longitude"])
        for row in validateList:
            writer.writerow(row)

    end_time = time.time()
    cost_time = (end_time - start_time)

    print("The number of items:" + str(itemsCount))
    print("The number of invalid items:" + str(invalidItemNumber))
    print('Total time spent on loading POI data {:.5f} second.'.format(cost_time))
    print("--------Done!--------")

# top_left = [-73.973205, 40.806470]
# bottom_left = [-74.035690, 40.709729]
# bottom_right = [-73.992431, 40.696715]
# top_right = [-73.934066, 40.781518]
def judgerOfGps(longitude,latitude):
    top_left = [-73.939961, 40.860598]
    bottom_left = [-74.035471, 40.709885]
    bottom_right = [-73.984962, 40.694339]
    top_right = [-73.926392, 40.801619]
    coordinates = [(longitude, latitude)]
    coordinates_shapely = MultiPoint(coordinates)
    polyNY_shapely = Polygon([(top_left), (bottom_left), (bottom_right), (top_right)])
    for pp in range(len(list(coordinates_shapely))):
        b=(polyNY_shapely.contains(coordinates_shapely[pp]))
    return b


filePath = "E:/data/ExprimentField/PoIs_NYC.txt"
getPOIFromCSVtoGraph(filePath)

