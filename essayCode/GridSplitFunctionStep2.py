
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

    locationIDToPOIMap = {}
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

